#pragma once
#include "common.hpp"
#include <cublas_v2.h>

namespace strata::qwen36_bf16 {
// BF16 projection boundary used by HF nn.Linear. Expert down projections which
// intentionally return FP32 continue to use Strata's prefill::Gemm.
class Bf16Linear {
public:
    ~Bf16Linear() {if(handle_) cublasDestroy(handle_);}
    bool init(cudaStream_t stream,std::string& error) {
        auto check=[&](cublasStatus_t status,const char* operation) {
            if(status==CUBLAS_STATUS_SUCCESS) return true;
            error=std::string(operation)+": cuBLAS status "+std::to_string(int(status));return false;
        };
        if(!check(cublasCreate(&handle_),"BF16 linear create")) return false;
        if(!check(cublasSetStream(handle_,stream),"BF16 linear stream")) return false;
        workspace_.allocate(32u<<20);
        if(!check(cublasSetWorkspace(handle_,workspace_.p,workspace_.bytes),"BF16 linear workspace")) return false;
        return check(cublasSetMathMode(handle_,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION),"BF16 linear FP32 reduction");
    }
    void run(const BF* input,const BF* weight,BF* output,int rows,int columns,int inner) {
        const float alpha=1,beta=0;
        const auto status=cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,columns,rows,inner,&alpha,
            weight,CUDA_R_16BF,inner,input,CUDA_R_16BF,inner,&beta,output,CUDA_R_16BF,columns,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        require(status==CUBLAS_STATUS_SUCCESS,"BF16 output linear failed, cuBLAS status "+std::to_string(int(status)));
    }
private:
    cublasHandle_t handle_=nullptr;
    DeviceBuffer workspace_;
};
}
