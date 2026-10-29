#include "expert.hpp"
#include <map>

namespace strata::qwen36_bf16 {
namespace {
__global__ void gather(const BF* x,BF* out,const int* rows,int h,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n*h) out[i]=x[rows[i/h]*h+i%h];
}
__global__ void swiglu(const float* gu,BF* out,int ff,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n*ff) {
        int t=i/ff,j=i%ff;
        // Qwen HF rounds gate/up, SiLU, multiply, down, route weighting and
        // each expert addition to BF16; FP32 GEMM accumulation stays intact.
        float g=__bfloat162float(__float2bfloat16_rn(gu[t*2*ff+j]));
        float u=__bfloat162float(__float2bfloat16_rn(gu[t*2*ff+ff+j]));
        float activated=__bfloat162float(__float2bfloat16_rn(g/(1.0f+expf(-g))));
        out[i]=__float2bfloat16_rn(activated*u);
    }
}
__global__ void combine(const float* down,float* sum,const int* rows,const float* weights,int h,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n*h) {
        int t=i/h,j=i%h;
        float projected=__bfloat162float(__float2bfloat16_rn(down[i]));
        float routed=__bfloat162float(__float2bfloat16_rn(projected*weights[t]));
        const int at=rows[t]*h+j;
        sum[at]=__bfloat162float(__float2bfloat16_rn(sum[at]+routed));
    }
}
__global__ void round_output(const float* sum,BF* out,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n) out[i]=__float2bfloat16_rn(sum[i]);
}
__global__ void gather_slots(const BF* x,BF* out,const int* slot_rows,int h,int slots) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(slots)*h) return;
    const int slot=int(i/h),src=slot_rows[slot];if(src>=0) out[i]=x[size_t(src)*h+i%h];
}
// Per token, contributions are added in ascending expert id with the same
// BF16 rounding after every addition as the per-group combine() loop.
__global__ void combine_tokens(const float* down,const int* token_slots,const float* token_weights,BF* out,int h,int tokens,int top_k) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=tokens*h) return;
    const int t=i/h,j=i%h;float sum=0;
    for(int k=0;k<top_k;++k) {
        const int slot=token_slots[t*top_k+k];
        const float projected=__bfloat162float(__float2bfloat16_rn(down[size_t(slot)*h+j]));
        const float routed=__bfloat162float(__float2bfloat16_rn(projected*token_weights[t*top_k+k]));
        sum=__bfloat162float(__float2bfloat16_rn(sum+routed));
    }
    out[i]=__float2bfloat16_rn(sum);
}
}
ExpertExecutor::ExpertExecutor(ExpertStore& store,int max_tokens,cudaStream_t stream)
    :store_(store),max_tokens_(32),stream_(stream),
     gathered_(size_t(32)*store.hidden*2),gate_up_(size_t(32)*2*store.ff*4),
     hidden_(size_t(32)*store.ff*2),down_(size_t(32)*store.hidden*4),
     sum_(size_t(32)*store.hidden*4),row_ids_(size_t(32)*store.experts*4),row_weights_(size_t(32)*store.experts*4) {
    Q36_CUDA(cudaMallocHost(&host_rows_,size_t(32)*store.experts*4));
    Q36_CUDA(cudaMallocHost(&host_scales_,size_t(32)*store.experts*4));
    Q36_CUDA(cudaEventCreateWithFlags(&table_done_,cudaEventDisableTiming));
    Q36_CUDA(cudaMemsetAsync(gathered_.p,0,gathered_.bytes,stream));
    require(max_tokens>0 && max_tokens<=32,"Initial executor supports at most 32 actual rows");
    const char* forced=std::getenv("STRATA_BF16_TC");
    require(!forced||std::atoi(forced)!=2,"STRATA_BF16_TC=2 forces FP16 conversion; unset it for the Qwen3.6 BF16 BF16 baseline");
    int device=0;cudaDeviceProp properties{};Q36_CUDA(cudaGetDevice(&device));Q36_CUDA(cudaGetDeviceProperties(&properties,device));
    require(properties.major>=8,"Qwen3.6 BF16 BF16 baseline requires native BF16 CUDA hardware (compute capability >=8.0)");
    std::string error;require(gemm_.init(stream,0,error),error);
    if(store.resident()) {
        max_groups_=store.experts;const int slots=max_groups_*32;
        require(cublasCreate(&batch_handle_)==CUBLAS_STATUS_SUCCESS,"cuBLAS create failed");
        require(cublasSetStream(batch_handle_,stream)==CUBLAS_STATUS_SUCCESS,"cuBLAS stream failed");
        batch_ws_.allocate(32u<<20);
        require(cublasSetWorkspace(batch_handle_,batch_ws_.p,batch_ws_.bytes)==CUBLAS_STATUS_SUCCESS,"cuBLAS workspace failed");
        require(cublasSetMathMode(batch_handle_,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)==CUBLAS_STATUS_SUCCESS,"cuBLAS math mode failed");
        b_gathered_.allocate(size_t(slots)*store.hidden*2);b_gate_up_.allocate(size_t(slots)*2*store.ff*4);
        b_hidden_.allocate(size_t(slots)*store.ff*2);b_down_.allocate(size_t(slots)*store.hidden*4);
        Q36_CUDA(cudaMemsetAsync(b_gathered_.p,0,b_gathered_.bytes,stream));
        // table: 6 pointer arrays [groups] + slot rows [slots] + token slots/weights [32*top_k<=32*experts]
        b_host_bytes_=size_t(max_groups_)*6*8+size_t(slots)*4+size_t(32)*store.experts*8;
        Q36_CUDA(cudaMallocHost(&b_host_,b_host_bytes_));b_table_.allocate(b_host_bytes_);
    }
}
void ExpertExecutor::run_batched(int layer,const BF* x,BF* output,int tokens,int top_k,
                                 const std::map<int,std::vector<std::pair<int,float>>>& groups) {
    const int G=int(groups.size()),h=store_.hidden,ff=store_.ff;
    require(G>0&&G<=max_groups_,"Batched expert group count out of range");
    Q36_CUDA(cudaEventSynchronize(table_done_));
    auto* host=static_cast<uint8_t*>(b_host_);
    void** a_gu=reinterpret_cast<void**>(host);void** b_gu=a_gu+G;void** c_gu=b_gu+G;
    void** a_dn=c_gu+G;void** b_dn=a_dn+G;void** c_dn=b_dn+G;
    int* slot_rows=reinterpret_cast<int*>(c_dn+G);int* token_slots=slot_rows+G*32;float* token_weights=reinterpret_cast<float*>(token_slots+tokens*top_k);
    std::vector<int> filled(size_t(tokens),0);int g=0;
    for(const auto& group:groups) {
        const BF* blob=store_.acquire(layer,group.first);store_.used(layer,group.first);
        a_gu[g]=const_cast<BF*>(blob);b_gu[g]=b_gathered_.as<BF>()+size_t(g)*32*h;c_gu[g]=b_gate_up_.as<float>()+size_t(g)*32*2*ff;
        a_dn[g]=const_cast<BF*>(blob)+store_.gu_bytes/2;b_dn[g]=b_hidden_.as<BF>()+size_t(g)*32*ff;c_dn[g]=b_down_.as<float>()+size_t(g)*32*h;
        int r=0;
        for(const auto& row:group.second) {
            slot_rows[g*32+r]=row.first;const int t=row.first,k=filled[size_t(t)]++;
            token_slots[t*top_k+k]=g*32+r;token_weights[t*top_k+k]=row.second;++r;
        }
        for(;r<32;++r) slot_rows[g*32+r]=-1;
        ++g;
    }
    for(int t=0;t<tokens;++t) require(filled[size_t(t)]==top_k,"Every token needs top_k routed contributions");
    const size_t bytes=reinterpret_cast<uint8_t*>(token_weights+tokens*top_k)-host;
    Q36_CUDA(cudaMemcpyAsync(b_table_.p,host,bytes,cudaMemcpyHostToDevice,stream_));
    Q36_CUDA(cudaEventRecord(table_done_,stream_));
    auto* dev=b_table_.as<uint8_t>();
    auto dptr=[&](const void* hp){return dev+(static_cast<const uint8_t*>(hp)-host);};
    gather_slots<<<unsigned((size_t(G)*32*h+255)/256),256,0,stream_>>>(x,b_gathered_.as<BF>(),reinterpret_cast<const int*>(dptr(slot_rows)),h,G*32);
    const float alpha=1,beta=0;
    // Same FP32-accumulated 32-row geometry per expert as prefill::Gemm::bf16.
    auto st=cublasGemmBatchedEx(batch_handle_,CUBLAS_OP_T,CUBLAS_OP_N,2*ff,32,h,&alpha,
        reinterpret_cast<const void* const*>(dptr(a_gu)),CUDA_R_16BF,h,reinterpret_cast<const void* const*>(dptr(b_gu)),CUDA_R_16BF,h,
        &beta,reinterpret_cast<void* const*>(dptr(c_gu)),CUDA_R_32F,2*ff,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    require(st==CUBLAS_STATUS_SUCCESS,"Batched gate/up GEMM failed: "+std::to_string(int(st)));
    swiglu<<<unsigned((size_t(G)*32*ff+255)/256),256,0,stream_>>>(b_gate_up_.as<float>(),b_hidden_.as<BF>(),ff,G*32);
    st=cublasGemmBatchedEx(batch_handle_,CUBLAS_OP_T,CUBLAS_OP_N,h,32,ff,&alpha,
        reinterpret_cast<const void* const*>(dptr(a_dn)),CUDA_R_16BF,ff,reinterpret_cast<const void* const*>(dptr(b_dn)),CUDA_R_16BF,ff,
        &beta,reinterpret_cast<void* const*>(dptr(c_dn)),CUDA_R_32F,h,G,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    require(st==CUBLAS_STATUS_SUCCESS,"Batched down GEMM failed: "+std::to_string(int(st)));
    combine_tokens<<<(tokens*h+255)/256,256,0,stream_>>>(b_down_.as<float>(),reinterpret_cast<const int*>(dptr(token_slots)),
        reinterpret_cast<const float*>(dptr(token_weights)),output,h,tokens,top_k);
    Q36_CUDA(cudaGetLastError());
}
void ExpertExecutor::run(int layer,const BF* x,BF* output,int tokens,int top_k,
                         const std::vector<int>& ids,const std::vector<float>& weights) {
    require(tokens>0&&tokens<=max_tokens_ && top_k>0&&top_k<=store_.experts,"Invalid expert batch dimensions");
    require(ids.size()==size_t(tokens)*top_k && ids.size()==weights.size(),"Wrong route array size");
    std::map<int,std::vector<std::pair<int,float>>> groups;
    for(int t=0;t<tokens;++t) {
        std::vector<bool> seen(size_t(store_.experts),false);
        for(int k=0;k<top_k;++k) {
            const size_t at=size_t(t)*top_k+k;const int e=ids[at];
            require(e>=0&&e<store_.experts&&!seen[size_t(e)],"Duplicate/out-of-range route expert");
            require(std::isfinite(weights[at])&&weights[at]>=0,"Invalid route weight");seen[size_t(e)]=true;
            groups[e].emplace_back(t,weights[at]);
        }
    }
    const char* batched_env=std::getenv("STRATA_Q36_BATCHED_EXPERTS");
    if(batch_handle_&&!(batched_env&&std::atoi(batched_env)==0)) {run_batched(layer,x,output,tokens,top_k,groups);return;}
    Q36_CUDA(cudaMemsetAsync(sum_.p,0,size_t(tokens)*store_.hidden*4,stream_));
    // The previous call's table copy must finish before the pinned table is
    // rewritten; one H2D per call replaces two synchronous copies per expert.
    Q36_CUDA(cudaEventSynchronize(table_done_));
    std::vector<int> offsets;offsets.reserve(groups.size());int packed=0;
    for(const auto& group:groups) {
        offsets.push_back(packed);
        for(const auto& r:group.second) {host_rows_[packed]=r.first;host_scales_[packed]=r.second;++packed;}
    }
    Q36_CUDA(cudaMemcpyAsync(row_ids_.p,host_rows_,size_t(packed)*4,cudaMemcpyHostToDevice,stream_));
    Q36_CUDA(cudaMemcpyAsync(row_weights_.p,host_scales_,size_t(packed)*4,cudaMemcpyHostToDevice,stream_));
    Q36_CUDA(cudaEventRecord(table_done_,stream_));
    size_t gi=0;
    for(const auto& group:groups) {
        const int e=group.first,n=int(group.second.size());
        const int* rows_d=row_ids_.as<int>()+offsets[gi];const float* scales_d=row_weights_.as<float>()+offsets[gi];++gi;
        const BF* blob=store_.acquire(layer,e);
        // Rows >= n of the fixed 32-row GEMM hold stale activations; GEMM rows
        // are independent and combine() reads only the first n outputs.
        gather<<<(n*store_.hidden+255)/256,256,0,stream_>>>(x,gathered_.as<BF>(),rows_d,store_.hidden,n);
        gemm_.bf16(gathered_.as<uint16_t>(),reinterpret_cast<const uint16_t*>(blob),gate_up_.as<float>(),max_tokens_,2*store_.ff,store_.hidden);
        swiglu<<<(max_tokens_*store_.ff+255)/256,256,0,stream_>>>(gate_up_.as<float>(),hidden_.as<BF>(),store_.ff,max_tokens_);
        gemm_.bf16(hidden_.as<uint16_t>(),reinterpret_cast<const uint16_t*>(blob)+store_.gu_bytes/2,
                   down_.as<float>(),max_tokens_,store_.hidden,store_.ff);
        combine<<<(n*store_.hidden+255)/256,256,0,stream_>>>(down_.as<float>(),sum_.as<float>(),rows_d,scales_d,store_.hidden,n);
        Q36_CUDA(cudaGetLastError());store_.used(layer,e);
    }
    round_output<<<(tokens*store_.hidden+255)/256,256,0,stream_>>>(sum_.as<float>(),output,tokens*store_.hidden);
    Q36_CUDA(cudaGetLastError());
}
ExpertExecutor::~ExpertExecutor() {
    if(batch_handle_) cublasDestroy(batch_handle_);
    if(b_host_) cudaFreeHost(b_host_);
    if(table_done_) {cudaEventSynchronize(table_done_);cudaEventDestroy(table_done_);}
    if(host_rows_) cudaFreeHost(host_rows_);
    if(host_scales_) cudaFreeHost(host_scales_);
}
Json ExpertExecutor::workspace() const {
    return {{"max_tokens",max_tokens_},{"gemm_compute_rows",max_tokens_},{"device_activation_bytes",gathered_.bytes+gate_up_.bytes+hidden_.bytes+down_.bytes+sum_.bytes+row_ids_.bytes+row_weights_.bytes},
            {"strata_gemm_workspace_bytes",32u<<20},{"weight_dtype","BF16"},{"gemm_accumulator","FP32"}};
}
} // namespace strata::qwen36_bf16
