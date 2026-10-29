#pragma once
#include "store.hpp"
#include "strata/prefill/gemm.hpp"
#include <cublas_v2.h>
#include <map>

namespace strata::qwen36_bf16 {
class ExpertExecutor {
public:
    ExpertExecutor(ExpertStore& store,int max_tokens,cudaStream_t stream);
    // ids/weights are row-major [tokens,top_k]. Contributions are rounded to
    // BF16 before the FP32 expert sum, in ascending expert id (HF reference).
    void run(int layer,const BF* x,BF* output,int tokens,int top_k,
             const std::vector<int>& ids,const std::vector<float>& weights);
    Json workspace() const;
private:
    ExpertStore& store_;
    int max_tokens_;
    cudaStream_t stream_;
    prefill::Gemm gemm_;
    DeviceBuffer gathered_,gate_up_,hidden_,down_,sum_,row_ids_,row_weights_;
    // Pinned per-call route table: every group's rows/scales go up in one copy.
    int* host_rows_=nullptr;float* host_scales_=nullptr;cudaEvent_t table_done_=nullptr;
    // Resident-store path: one batched cuBLAS GEMM per projection per layer.
    void run_batched(int layer,const BF* x,BF* output,int tokens,int top_k,
                     const std::map<int,std::vector<std::pair<int,float>>>& groups);
    cublasHandle_t batch_handle_=nullptr;DeviceBuffer batch_ws_;
    DeviceBuffer b_gathered_,b_gate_up_,b_hidden_,b_down_,b_table_;
    void* b_host_=nullptr;size_t b_host_bytes_=0;int max_groups_=0;
public:
    ~ExpertExecutor();
};
} // namespace strata::qwen36_bf16
