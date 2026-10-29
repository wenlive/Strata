#pragma once
#include "expert.hpp"
#include "kernels.hpp"
#include "router.hpp"

namespace strata::qwen36_bf16 {
struct ForwardResult {std::vector<float> logits;Json measurement;};
class Model {
public:
    Model(ExpertStore& source,int context,int max_tokens,cudaStream_t stream,bool layer_test=false);
    void reset();
    ForwardResult forward(const std::vector<int>& tokens,bool preview,bool commit,const std::string& phase);
    void save_state(const std::filesystem::path& root);
    uint64_t state_hash();
    Json allocation() const;
    void trace_to(const std::filesystem::path& root) {trace_root_=root;}
    int prefix=0,vocab=0,block_size=0;
    Json config;
private:
    struct State {bool linear=false;DeviceBuffer conv,ssm,key,value;};
    ExpertStore& store_;
    bool layer_test_=false;
    int context_=0,max_tokens_=0,h_=0,ff_=0,kh_=0,vh_=0,kd_=0,vd_=0,ch_=0,nh_=0,nkv_=0,hd_=0,sf_=0,top_k_=0;
    cudaStream_t stream_;
    std::map<std::string,DeviceBuffer> weights_;
    std::vector<State> state_;
    std::filesystem::path trace_root_;
    prefill::Gemm dense_;
    ExpertExecutor experts_;
    DeviceBuffer x_,xn_,mixer_,expert_out_,shared_gate_,shared_up_,shared_mid_,shared_out_,shared_scale_,router_,ids_,fp32_,logits_,dense_input_;
    DeviceBuffer raw_,mixed_,qf_,kf_,z_,a_,b_,g_,beta_,features_,features_norm_,qatt_,knew_,katt_,vnew_;
    const BF* weight(const std::string& name) const;
    void linear(const std::string& name,const BF* x,BF* y,int tokens);
};
Json counter_delta(const Json& after,const Json& before);
}
