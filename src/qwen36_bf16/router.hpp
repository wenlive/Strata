#pragma once
#include "common.hpp"
#include <numeric>

namespace strata::qwen36_bf16 {
struct Routes {
    std::vector<int> ids;
    std::vector<float> weights;
    std::vector<int> coreset;
};
inline Routes route(const std::vector<float>& logits,int tokens,int experts,int top_k,bool preview,float top_p=0.9f) {
    require(tokens>0 && experts>0 && top_k>0&&top_k<=experts && logits.size()==size_t(tokens)*experts,"Invalid router dimensions");
    require(top_p>0&&top_p<=1,"Invalid coreset top-p");
    for(float x:logits) require(std::isfinite(x),"Nonfinite router logit");
    auto order_row=[&](const float* row) {
        std::vector<int> ids(size_t(experts),0);std::iota(ids.begin(),ids.end(),0);
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return row[a]>row[b];});return ids;
    };
    std::vector<float> adjusted=logits;
    Routes result;
    if(preview) {
        std::vector<float> totals(size_t(experts),0);
        for(int t=0;t<tokens;++t) {
            const float* row=logits.data()+size_t(t)*experts;auto ids=order_row(row);
            float denom=0;for(int e=0;e<experts;++e) denom+=std::exp(row[e]-row[ids[0]]);
            for(int k=0;k<top_k;++k) totals[size_t(ids[size_t(k)])]+=std::exp(row[ids[size_t(k)]]-row[ids[0]])/denom;
        }
        float total=0;for(float x:totals) total+=x;total=std::max(total,1e-9f);
        std::vector<int> order(size_t(experts),0);std::iota(order.begin(),order.end(),0);
        for(float& x:totals) x/=total;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return totals[size_t(a)]>totals[size_t(b)];});
        std::vector<bool> allowed(size_t(experts),false);float cumulative=0;
        for(int i=0;i<experts;++i) {
            const int e=order[size_t(i)];const float p=totals[size_t(e)];cumulative+=p;
            if(i==0 || cumulative-p<top_p) {allowed[size_t(e)]=true;result.coreset.push_back(e);}
        }
        for(int t=0;t<tokens;++t) for(int e=0;e<experts;++e)
            if(!allowed[size_t(e)]) adjusted[size_t(t)*experts+e]=-std::numeric_limits<float>::infinity();
    }
    result.ids.reserve(size_t(tokens)*top_k);result.weights.reserve(size_t(tokens)*top_k);
    for(int t=0;t<tokens;++t) {
        const float* row=adjusted.data()+size_t(t)*experts;auto ids=order_row(row);
        float denom=0;for(int e=0;e<experts;++e) denom+=std::exp(row[e]-row[ids[0]]);
        std::vector<float> w(size_t(top_k),0);float sum=0;
        for(int k=0;k<top_k;++k) {w[size_t(k)]=std::exp(row[ids[size_t(k)]]-row[ids[0]])/denom;sum+=w[size_t(k)];}
        sum=std::max(sum,1e-12f);
        for(int k=0;k<top_k;++k) {result.ids.push_back(ids[size_t(k)]);result.weights.push_back(w[size_t(k)]/sum);}
    }
    return result;
}
}
