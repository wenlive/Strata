#pragma once
#include "common.hpp"
#include <set>

namespace strata::clover {
inline int greedy_argmax(const std::vector<float>& logits,int row,int vocab) {
    require(vocab>0&&row>=0&&size_t(row+1)*vocab<=logits.size(),"Argmax shape mismatch");
    int best=0;float score=-std::numeric_limits<float>::infinity();
    for(int v=0;v<vocab;++v) {
        const float x=logits[size_t(row)*vocab+v];require(!std::isnan(x),"NaN final logit");
        if(x>score) {score=x;best=v;}
    }
    require(std::isfinite(score),"No finite token logit");return best;
}
struct ExactDecision {
    int matched_drafts=0,bonus=-1,emitted_drafts=0,pending=-1;
    bool emitted_bonus=false;
    std::vector<int> output,commit_inputs;
    Json json() const {return {{"matched_drafts",matched_drafts},{"bonus",bonus},
        {"emitted_drafts",emitted_drafts},{"emitted_bonus",emitted_bonus},{"pending",pending},
        {"output",output},{"commit_inputs",commit_inputs}};}
};
// candidates[0] is already emitted but remains pending/uncommitted. The causal
// logit at j verifies candidate j+1. The newly sampled boundary bonus is output
// once and remains uncommitted. An EOS/length-truncated accepted prefix is the
// only prefix replayed; no rejected or unreturned draft is persisted.
// argmax[j] is the greedy token of verify row j (penalty-adjusted).
inline ExactDecision decide_exact_ids(const std::vector<int>& candidates,const std::vector<int>& argmax,
                                      int remaining,const std::set<int>& eos) {
    require(candidates.size()>=2&&remaining>0&&argmax.size()==candidates.size(),"Invalid exact decision shape");
    ExactDecision result;
    const int width=int(candidates.size());
    while(result.matched_drafts<width-1&&candidates[size_t(result.matched_drafts+1)]==argmax[size_t(result.matched_drafts)]) ++result.matched_drafts;
    result.bonus=argmax[size_t(result.matched_drafts)];
    for(int pos=0;pos<=result.matched_drafts&&int(result.output.size())<remaining;++pos) {
        const int token=pos<result.matched_drafts?candidates[size_t(pos+1)]:result.bonus;
        result.output.push_back(token);
        if(pos<result.matched_drafts) ++result.emitted_drafts;else result.emitted_bonus=true;
        if(eos.count(token)) break;
    }
    result.commit_inputs.assign(candidates.begin(),candidates.begin()+1+result.emitted_drafts);
    result.pending=result.emitted_bonus?result.bonus:-1;return result;
}
inline ExactDecision decide_exact(const std::vector<int>& candidates,const std::vector<float>& logits,
                                  int vocab,int remaining,const std::set<int>& eos) {
    require(candidates.size()>=2&&remaining>0&&logits.size()==candidates.size()*size_t(vocab),"Invalid exact decision shape");
    std::vector<int> argmax(candidates.size());
    for(size_t j=0;j<candidates.size();++j) argmax[j]=greedy_argmax(logits,int(j),vocab);
    return decide_exact_ids(candidates,argmax,remaining,eos);
}
}
