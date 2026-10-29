#pragma once
#include "speculation.hpp"
#include <array>

namespace strata::clover {
struct DenoiseDecision {
    std::vector<int> prediction, transfer;
    std::vector<float> confidence;
};
// HF LowConfidence: confidence is T=1 softmax of fresh, penalized FP32
// logits. Strict > threshold; if none qualify, resolve the most confident
// pending position, choosing the first position on a tie. Resolved positions
// remain in the next full preview but cannot be overwritten.
inline DenoiseDecision decide_denoise(const std::vector<float>& logits,
                                    const std::vector<bool>& pending,int vocab,float threshold) {
    require(vocab>0&&!pending.empty()&&logits.size()==pending.size()*size_t(vocab),"Invalid denoising logits shape");
    require(std::isfinite(threshold)&&threshold>=0&&threshold<=1,"Denoising threshold must be finite and in [0,1]");
    DenoiseDecision out;int fallback=-1;
    for(size_t row=0;row<pending.size();++row) {
        const int best=greedy_argmax(logits,int(row),vocab);out.prediction.push_back(best);
        const float maximum=logits[row*size_t(vocab)+best];
        // Bounded compensated FP32 partial sums: adding many small tails to
        // a dominant logit with a plain serial sum loses confidence near .95.
        // Reduction order may differ from Torch; the oracle reports its error.
        std::array<float,256> sums{},correction{};
        for(int v=0;v<vocab;++v) {
            const size_t lane=size_t(v)&255;
            const float value=expf(logits[row*size_t(vocab)+v]-maximum)-correction[lane];
            const float next=sums[lane]+value;
            correction[lane]=(next-sums[lane])-value;sums[lane]=next;
        }
        for(int stride=128;stride;stride/=2) for(int j=0;j<stride;++j) sums[size_t(j)]+=sums[size_t(j+stride)];
        const float confidence=1.0f/sums[0];require(std::isfinite(confidence),"Non-finite denoising confidence");
        out.confidence.push_back(confidence);
        if(!pending[row]) continue;
        if(fallback<0||confidence>out.confidence[size_t(fallback)]) fallback=int(row);
        if(confidence>threshold) out.transfer.push_back(int(row));
    }
    require(fallback>=0,"Denoising step has no pending positions");
    if(out.transfer.empty()) out.transfer.push_back(fallback);
    return out;
}
inline std::vector<int> denoise_output(const std::vector<int>& block,int remaining,const std::set<int>& stops) {
    require(remaining>0&&!block.empty(),"Invalid denoising output limit");
    std::vector<int> output;
    for(int token:block) {output.push_back(token);if(int(output.size())==remaining||stops.count(token)) break;}
    return output;
}
}
