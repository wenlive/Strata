#pragma once
#include "denoising.hpp"

namespace strata::clover {
struct Penalties {
    float presence=0,frequency=0,repetition=1,length=0,think=0;
    int length_start=8192,length_window=8192,think_start=4096,think_window=4096,think_token=248069;
    std::vector<int> counts;std::set<int> stops;int output_length=0;
    explicit Penalties(const Json& gen,const Json& cfg,int vocab) :counts(size_t(vocab),0) {
        presence=cfg.value("presence_penalty",gen.value("presence_penalty",0.0f));
        frequency=cfg.value("frequency_penalty",gen.value("frequency_penalty",0.0f));
        repetition=cfg.value("repetition_penalty",gen.value("repetition_penalty",1.0f));
        require(std::isfinite(presence)&&std::abs(presence)<=2&&std::isfinite(frequency)&&std::abs(frequency)<=2&&std::isfinite(repetition)&&repetition>0&&repetition<=2,"Invalid sampling penalties");
        length=gen.value("dllm_length_penalty",0.0f);think=gen.value("dllm_think_end_bias",0.0f);
        length_start=gen.value("dllm_length_penalty_start",8192);length_window=gen.value("dllm_length_penalty_window",8192);
        think_start=gen.value("dllm_think_end_start",4096);think_window=gen.value("dllm_think_end_window",4096);
        think_token=gen.value("dllm_think_end_token_id",248069);
        require(length_window>0&&think_window>0&&length_start>=0&&think_start>=0&&think_token>=0&&think_token<vocab,"Invalid length/thinking penalty configuration");
        const auto& eos=gen.at("eos_token_id");
        if(eos.is_array()) for(const auto& token:eos) stops.insert(token.get<int>());
        else stops.insert(eos.get<int>());
        for(int token:stops) require(token>=0&&token<vocab,"Invalid EOS id");
    }
    void commit(int token) {++counts[size_t(token)];++output_length;}
    void apply(std::vector<float>& logits,int rows,int vocab,bool preview,const std::vector<int>& visible,bool custom=true) const {
        for(int row=0;row<rows;++row) for(int token=0;token<vocab;++token) {
            float& value=logits[size_t(row)*vocab+token];int count=counts[size_t(token)];
            if(frequency) value-=float(count)*frequency;
            if(presence&&count) value-=presence;
            if(repetition!=1&&count) value=value<0?value*repetition:value/repetition;
        }
        if(!custom) return;
        const int first=output_length+(preview?0:1);
        if(length) for(int row=0;row<rows;++row) {
            float ramp=length*std::clamp(float(first+row-length_start)/float(length_window),0.0f,1.0f);
            for(int token:stops) logits[size_t(row)*vocab+token]+=ramp;
        }
        bool close_visible=std::find(visible.begin(),visible.end(),think_token)!=visible.end();
        if(think&&!counts[size_t(think_token)]&&(!preview||!close_visible)) {
            const int row=std::max(int(preview),think_start+1-first);
            if(row<rows) logits[size_t(row)*vocab+think_token]+=think*std::min(float(first+row-think_start)/float(think_window),1.0f);
        }
    }
    void apply_denoise(std::vector<float>& logits,int vocab,const std::vector<int>& block,int mask) const {
        const bool visible_close=std::find(block.begin(),block.end(),think_token)!=block.end();
        for(size_t row=0;row<block.size();++row) {
            logits[row*size_t(vocab)+mask]=-std::numeric_limits<float>::infinity();
            for(int token=0;token<vocab;++token) {
                float& value=logits[row*size_t(vocab)+token];const int count=counts[size_t(token)];
                if(frequency) value-=float(count)*frequency;
                if(presence&&count) value-=presence;
                if(repetition!=1&&count) value=value<0?value*repetition:value/repetition;
            }
            const int position=output_length+int(row)+1;
            if(length) for(int token:stops) logits[row*size_t(vocab)+token]+=length*std::clamp(float(position-length_start)/float(length_window),0.0f,1.0f);
            // HF block denoising biases every row until a closing token is
            // visible anywhere in this block or in committed response history.
            if(think&&!counts[size_t(think_token)]&&!visible_close)
                logits[row*size_t(vocab)+think_token]+=think*std::clamp(float(position-think_start)/float(think_window),0.0f,1.0f);
        }
    }
};
}
