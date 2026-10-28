#pragma once
#include "store.hpp"
#include "strata/prefill/gemm.hpp"

namespace strata::clover {
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
};
} // namespace strata::clover
