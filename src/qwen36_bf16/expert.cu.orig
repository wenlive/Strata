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
}
ExpertExecutor::ExpertExecutor(ExpertStore& store,int max_tokens,cudaStream_t stream)
    :store_(store),max_tokens_(32),stream_(stream),
     gathered_(size_t(32)*store.hidden*2),gate_up_(size_t(32)*2*store.ff*4),
     hidden_(size_t(32)*store.ff*2),down_(size_t(32)*store.hidden*4),
     sum_(size_t(32)*store.hidden*4),row_ids_(size_t(32)*4),row_weights_(size_t(32)*4) {
    require(max_tokens>0 && max_tokens<=32,"Initial executor supports at most 32 actual rows");
    const char* forced=std::getenv("STRATA_BF16_TC");
    require(!forced||std::atoi(forced)!=2,"STRATA_BF16_TC=2 forces FP16 conversion; unset it for the Qwen3.6 BF16 BF16 baseline");
    int device=0;cudaDeviceProp properties{};Q36_CUDA(cudaGetDevice(&device));Q36_CUDA(cudaGetDeviceProperties(&properties,device));
    require(properties.major>=8,"Qwen3.6 BF16 BF16 baseline requires native BF16 CUDA hardware (compute capability >=8.0)");
    std::string error;require(gemm_.init(stream,0,error),error);
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
    Q36_CUDA(cudaMemsetAsync(sum_.p,0,size_t(tokens)*store_.hidden*4,stream_));
    for(const auto& group:groups) {
        const int e=group.first,n=int(group.second.size());
        std::vector<int> rows;std::vector<float> scales;rows.reserve(size_t(n));scales.reserve(size_t(n));
        for(const auto& r:group.second) {rows.push_back(r.first);scales.push_back(r.second);}
        // These tiny pageable copies are intentionally synchronous: the vectors
        // stay alive through completion and no unbounded transfer queue exists.
        Q36_CUDA(cudaStreamSynchronize(stream_));
        Q36_CUDA(cudaMemcpy(row_ids_.p,rows.data(),size_t(n)*4,cudaMemcpyHostToDevice));
        Q36_CUDA(cudaMemcpy(row_weights_.p,scales.data(),size_t(n)*4,cudaMemcpyHostToDevice));
        const BF* blob=store_.acquire(layer,e);
        Q36_CUDA(cudaMemsetAsync(gathered_.p,0,gathered_.bytes,stream_));
        gather<<<(n*store_.hidden+255)/256,256,0,stream_>>>(x,gathered_.as<BF>(),row_ids_.as<int>(),store_.hidden,n);
        gemm_.bf16(gathered_.as<uint16_t>(),reinterpret_cast<const uint16_t*>(blob),gate_up_.as<float>(),max_tokens_,2*store_.ff,store_.hidden);
        swiglu<<<(max_tokens_*store_.ff+255)/256,256,0,stream_>>>(gate_up_.as<float>(),hidden_.as<BF>(),store_.ff,max_tokens_);
        gemm_.bf16(hidden_.as<uint16_t>(),reinterpret_cast<const uint16_t*>(blob)+store_.gu_bytes/2,
                   down_.as<float>(),max_tokens_,store_.hidden,store_.ff);
        combine<<<(n*store_.hidden+255)/256,256,0,stream_>>>(down_.as<float>(),sum_.as<float>(),row_ids_.as<int>(),row_weights_.as<float>(),store_.hidden,n);
        Q36_CUDA(cudaGetLastError());store_.used(layer,e);
    }
    round_output<<<(tokens*store_.hidden+255)/256,256,0,stream_>>>(sum_.as<float>(),output,tokens*store_.hidden);
    Q36_CUDA(cudaGetLastError());
}
Json ExpertExecutor::workspace() const {
    return {{"max_tokens",max_tokens_},{"gemm_compute_rows",max_tokens_},{"device_activation_bytes",gathered_.bytes+gate_up_.bytes+hidden_.bytes+down_.bytes+sum_.bytes+row_ids_.bytes+row_weights_.bytes},
            {"strata_gemm_workspace_bytes",32u<<20},{"weight_dtype","BF16"},{"gemm_accumulator","FP32"}};
}
} // namespace strata::qwen36_bf16
