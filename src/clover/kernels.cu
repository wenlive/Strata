#include "kernels.hpp"
#include <cfloat>

namespace strata::clover::ops {
namespace {
__device__ float f(BF x) {return __bfloat162float(x);}
__device__ BF b(float x) {return __float2bfloat16_rn(x);}
__device__ float silu(float x) {return x/(1.0f+expf(-x));}
__device__ float reduce_sum(float x) {
    __shared__ float partial[32];
    for(int s=16;s;s/=2) x+=__shfl_down_sync(0xffffffff,x,s);
    if((threadIdx.x&31)==0) partial[threadIdx.x/32]=x;
    __syncthreads();
    float v=threadIdx.x<blockDim.x/32?partial[threadIdx.x]:0;
    if(threadIdx.x<32) for(int s=16;s;s/=2) v+=__shfl_down_sync(0xffffffff,v,s);
    if(threadIdx.x==0) partial[0]=v;
    __syncthreads();return partial[0];
}
__device__ float reduce_max(float x) {
    __shared__ float partial[32];
    for(int s=16;s;s/=2) x=fmaxf(x,__shfl_down_sync(0xffffffff,x,s));
    if((threadIdx.x&31)==0) partial[threadIdx.x/32]=x;
    __syncthreads();
    float v=threadIdx.x<blockDim.x/32?partial[threadIdx.x]:-FLT_MAX;
    if(threadIdx.x<32) for(int s=16;s;s/=2) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,s));
    if(threadIdx.x==0) partial[0]=v;
    __syncthreads();return partial[0];
}
__global__ void cast_kernel(const float* in,BF* out,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n) out[i]=b(in[i]);
}
__global__ void add_kernel(const BF* a,const BF* c,BF* out,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n) out[i]=b(f(a[i])+f(c[i]));
}
__global__ void norm_kernel(const BF* in,const BF* w,const BF* gate,BF* out,int width,float eps,bool centered) {
    const int row=int(blockIdx.x),j=int(threadIdx.x);float sum=0;
    for(int d=j;d<width;d+=int(blockDim.x)) {float x=f(in[row*width+d]);sum+=x*x;}
    float inv=rsqrtf(reduce_sum(sum)/float(width)+eps);
    for(int d=j;d<width;d+=int(blockDim.x)) {
        float value=f(in[row*width+d])*inv*(f(w[d])+(centered?1.0f:0.0f));
        if(gate) value*=silu(f(gate[row*width+d]));out[row*width+d]=b(value);
    }
}
__global__ void swiglu_kernel(const BF* gate,const BF* up,BF* out,int n) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n) out[i]=b(silu(f(gate[i]))*f(up[i]));
}
__global__ void sigmoid_kernel(const BF* in,const BF* gate,BF* out,int n,int width,bool scalar) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<n) {float g=f(gate[scalar?i/width:i]);float sigmoid=f(b(1.0f/(1.0f+expf(-g))));out[i]=b(f(in[i])*sigmoid);}
}
__global__ void embedding_kernel(const BF* table,const int* ids,BF* out,int n,int width) {
    int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n*width) out[i]=table[int64_t(ids[i/width])*width+i%width];
}
template<bool Gated>
__global__ void prepare_kernel(const BF* proj,const BF* weight,BF* out,BF* gate,int heads,int dim,int rd,int prefix,float theta,float eps) {
    extern __shared__ BF normalized[];
    const int row=int(blockIdx.x),t=row/heads,j=int(threadIdx.x);
    const int stride=Gated?2*dim:dim;float sum=0;
    for(int d=j;d<dim;d+=int(blockDim.x)) {float x=f(proj[row*stride+d]);sum+=x*x;}
    const float inv=rsqrtf(reduce_sum(sum)/float(dim)+eps);
    for(int d=j;d<dim;d+=int(blockDim.x)) {
        normalized[d]=b(f(proj[row*stride+d])*inv*(1.0f+f(weight[d])));
        if(Gated) gate[row*dim+d]=proj[row*stride+dim+d];
    }
    __syncthreads();
    for(int d=j;d<dim;d+=int(blockDim.x)) {
        if(d<rd) {
            const int pair=d%(rd/2);float freq=1.0f/powf(theta,float(2*pair)/float(rd));
            const float angle=float(prefix+t)*freq;
            const BF cosine=b(cosf(angle)),sine=b(sinf(angle));
            const BF first=normalized[pair],second=normalized[pair+rd/2];
            out[row*dim+d]=d<rd/2?b(f(b(f(first)*f(cosine)))-f(b(f(second)*f(sine))))
                                               :b(f(b(f(second)*f(cosine)))+f(b(f(first)*f(sine))));
        } else out[row*dim+d]=normalized[d];
    }
}
__global__ void attention_kernel(const BF* q,const BF* nk,const BF* nv,const BF* pk,const BF* pv,BF* out,int t,int prefix,int nh,int nkv,int dim,bool preview) {
    extern __shared__ float scores[];
    const int head=int(blockIdx.x)%nh,token=int(blockIdx.x)/nh,kv=head/(nh/nkv);
    const int count=prefix+(preview?t:token+1);
    float mx=-FLT_MAX;
    for(int s=int(threadIdx.x);s<count;s+=int(blockDim.x)) {
        const BF* key=s<prefix?pk+(s*nkv+kv)*dim:nk+((s-prefix)*nkv+kv)*dim;
        float score=0;for(int d=0;d<dim;++d) score+=f(q[(token*nh+head)*dim+d])*f(key[d]);
        score*=rsqrtf(float(dim));scores[s]=score;mx=fmaxf(mx,score);
    }
    mx=reduce_max(mx);float total=0;
    for(int s=int(threadIdx.x);s<count;s+=int(blockDim.x)) {float v=expf(scores[s]-mx);scores[s]=v;total+=v;}
    total=reduce_sum(total);
    for(int s=int(threadIdx.x);s<count;s+=int(blockDim.x)) scores[s]/=total;
    __syncthreads();
    for(int d=int(threadIdx.x);d<dim;d+=int(blockDim.x)) {
        float value=0;
        for(int s=0;s<count;++s) {const BF* v=s<prefix?pv+(s*nkv+kv)*dim:nv+((s-prefix)*nkv+kv)*dim;value+=scores[s]*f(v[d]);}
        out[(token*nh+head)*dim+d]=b(value);
    }
}
__global__ void conv_kernel(const BF* raw,const BF* hist,const BF* left,const BF* right,BF* out,int t,int channels,int taps,bool preview) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=t*channels) return;
    const int token=i/channels,c=i%channels;float value=0;
    for(int tap=0;tap<taps;++tap) {
        const int pos=token+tap-(taps-1);
        const BF x=pos<0?hist[c*(taps-1)+(taps-1)+pos]:raw[pos*channels+c];
        const float product=f(x)*f(left[c*taps+tap]);
        value+=preview?product:f(b(product));
    }
    if(preview) for(int distance=1;distance<taps && token+distance<t;++distance)
        value+=f(raw[(token+distance)*channels+c])*f(right[c*(taps-1)+distance-1]);
    out[i]=b(silu(value));
}
__global__ void commit_conv_kernel(const BF* raw,BF* history,int t,int channels,int taps) {
    int c=int(blockIdx.x*blockDim.x+threadIdx.x);if(c>=channels) return;
    BF old[8];for(int i=0;i<taps-1;++i) old[i]=history[c*(taps-1)+i];
    for(int i=0;i<taps-1;++i) {
        const int pos=t-(taps-1)+i;
        history[c*(taps-1)+i]=pos>=0?raw[pos*channels+c]:old[(taps-1)+pos];
    }
}
__global__ void qk_kernel(const BF* mixed,float* q,float* k,int kh,int vh,int kd,int vd) {
    const int row=int(blockIdx.x),token=row/kh,head=row%kh,d=int(threadIdx.x),channels=2*kh*kd+vh*vd;
    float qv=d<kd?f(mixed[token*channels+head*kd+d]):0;
    float kv=d<kd?f(mixed[token*channels+kh*kd+head*kd+d]):0;
    float qsum=reduce_sum(qv*qv);float ksum=reduce_sum(kv*kv);
    if(d<kd) {q[row*kd+d]=qv*rsqrtf(qsum+1e-6f);k[row*kd+d]=kv*rsqrtf(ksum+1e-6f);}
}
__global__ void factors_kernel(const BF* a,const BF* beta_in,const BF* A_log,const BF* dt,float* g,BF* beta,int n,int heads) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i>=n) return;
    const int h=i%heads;const float v=f(a[i])+f(dt[h]);
    const float softplus=v>20?v:log1pf(expf(v));g[i]=-expf(f(A_log[h]))*softplus;
    beta[i]=b(1.0f/(1.0f+expf(-f(beta_in[i]))));
}
__global__ void scan_kernel(const BF* mixed,const float* q,const float* k,const float* g,const BF* beta,const float* in,float* outstate,BF* out,int t,int kh,int vh,int kd,int vd) {
    const int head=int(blockIdx.x),v=int(threadIdx.x),keyhead=head/(vh/kh),channels=2*kh*kd+vh*vd;
    float state[128];
    for(int d=0;d<kd;++d) state[d]=in[(head*vd+v)*kd+d];
    for(int token=0;token<t;++token) {
        const float* query=q+(token*kh+keyhead)*kd;const float* key=k+(token*kh+keyhead)*kd;
        float decay=expf(g[token*vh+head]),remembered=0;
        for(int d=0;d<kd;++d) {state[d]*=decay;remembered+=state[d]*key[d];}
        const float value=f(mixed[token*channels+2*kh*kd+head*vd+v]);
        const float delta=(value-remembered)*f(beta[token*vh+head]);
        float y=0;
        for(int d=0;d<kd;++d) {state[d]+=delta*key[d];y+=state[d]*(query[d]*rsqrtf(float(kd)));}
        out[(token*vh+head)*vd+v]=b(y);
    }
    if(outstate) for(int d=0;d<kd;++d) outstate[(head*vd+v)*kd+d]=state[d];
}
__global__ void preview_kernel(const BF* mixed,const float* q,const float* k,const float* g,const BF* scale,const BF* bias,const BF* hc,const float* state,BF* out,int t,int kh,int vh,int kd,int vd,int checkpoint) {
    __shared__ float logits[129];
    const int head=int(blockIdx.x)%vh,token=int(blockIdx.x)/vh,v=int(threadIdx.x),keyhead=head/(vh/kh),channels=2*kh*kd+vh*vd;
    const float* query=q+(token*kh+keyhead)*kd;
    float logit=-FLT_MAX;
    if(v<t) {
        const float* key=k+(v*kh+keyhead)*kd;float dot=0;
        for(int d=0;d<kd;++d) dot+=query[d]*key[d];logit=dot*f(scale[head]);
    } else if(v==t) logit=f(bias[head])+f(hc[head])*logf(float(t)/float(checkpoint));
    if(v<=t) logits[v]=logit;
    float mx=reduce_max(logit);float value=v<=t?expf(logit-mx):0;
    float total=reduce_sum(value);if(v<=t) logits[v]=value/total;
    __syncthreads();
    if(v<vd) {
        float historical=0;
        for(int d=0;d<kd;++d) historical+=(query[d]*rsqrtf(float(kd)))*state[(head*vd+v)*kd+d];
        historical*=expf(g[token*vh+head]);
        float local=0;for(int s=0;s<t;++s) local+=logits[s]*f(mixed[s*channels+2*kh*kd+head*vd+v]);
        out[(token*vh+head)*vd+v]=b(logits[t]*historical+local);
    }
}
}
void cast(const float* x,BF* y,int n,cudaStream_t s) {cast_kernel<<<(n+255)/256,256,0,s>>>(x,y,n);}
void residual(const BF* a,const BF* c,BF* y,int n,cudaStream_t s) {add_kernel<<<(n+255)/256,256,0,s>>>(a,c,y,n);}
void norm(const BF* x,const BF* w,const BF* g,BF* y,int rows,int width,float eps,bool centered,cudaStream_t s) {norm_kernel<<<rows,256,0,s>>>(x,w,g,y,width,eps,centered);}
void swiglu(const BF* gate,const BF* up,BF* out,int n,cudaStream_t s) {swiglu_kernel<<<(n+255)/256,256,0,s>>>(gate,up,out,n);}
void sigmoid_mul(const BF* x,const BF* g,BF* y,int rows,int width,bool scalar,cudaStream_t s) {sigmoid_kernel<<<(rows*width+255)/256,256,0,s>>>(x,g,y,rows*width,width,scalar);}
void embedding(const BF* table,const int* ids,BF* y,int t,int width,cudaStream_t s) {embedding_kernel<<<(t*width+255)/256,256,0,s>>>(table,ids,y,t,width);}
void prepare_q(const BF* p,const BF* w,BF* q,BF* g,int t,int h,int d,int rd,int prefix,float theta,float eps,cudaStream_t s) {prepare_kernel<true><<<t*h,256,size_t(d)*2,s>>>(p,w,q,g,h,d,rd,prefix,theta,eps);}
void prepare_k(const BF* p,const BF* w,BF* k,int t,int h,int d,int rd,int prefix,float theta,float eps,cudaStream_t s) {prepare_kernel<false><<<t*h,256,size_t(d)*2,s>>>(p,w,k,nullptr,h,d,rd,prefix,theta,eps);}
void attention(const BF* q,const BF* nk,const BF* nv,const BF* pk,const BF* pv,BF* y,int t,int prefix,int h,int kv,int d,bool preview,cudaStream_t s) {
    require(t>0 && prefix>=0 && t+prefix<=8192 && h%kv==0,"Unsupported short-context attention shape");
    attention_kernel<<<t*h,256,size_t(t+prefix)*4,s>>>(q,nk,nv,pk,pv,y,t,prefix,h,kv,d,preview);
}
void gdn_conv(const BF* raw,const BF* history,const BF* left,const BF* right,BF* y,int t,int ch,int taps,bool preview,cudaStream_t s) {
    require(taps>=2&&taps<=8,"Unsupported convolution width");conv_kernel<<<(t*ch+255)/256,256,0,s>>>(raw,history,left,right,y,t,ch,taps,preview);
}
void gdn_commit_conv(const BF* raw,BF* history,int t,int ch,int taps,cudaStream_t s) {commit_conv_kernel<<<(ch+255)/256,256,0,s>>>(raw,history,t,ch,taps);}
void gdn_qk(const BF* mixed,float* q,float* k,int t,int kh,int vh,int kd,int vd,cudaStream_t s) {
    require(kd==128&&vd==128 && vh%kh==0,"Initial GDN kernel requires KD=VD=128 and divisible heads");qk_kernel<<<t*kh,128,0,s>>>(mixed,q,k,kh,vh,kd,vd);
}
void gdn_factors(const BF* a,const BF* beta_in,const BF* al,const BF* dt,float* g,BF* beta,int t,int h,cudaStream_t s) {factors_kernel<<<(t*h+255)/256,256,0,s>>>(a,beta_in,al,dt,g,beta,t*h,h);}
void gdn_scan(const BF* mixed,const float* q,const float* k,const float* g,const BF* beta,const float* si,float* so,BF* y,int t,int kh,int vh,int kd,int vd,cudaStream_t s) {
    require(kd==128&&vd==128,"Unsupported GDN scan dimensions");scan_kernel<<<vh,128,0,s>>>(mixed,q,k,g,beta,si,so,y,t,kh,vh,kd,vd);
}
void gdn_preview(const BF* mixed,const float* q,const float* k,const float* g,const BF* scale,const BF* bias,const BF* hc,const float* state,BF* y,int t,int kh,int vh,int kd,int vd,int checkpoint,cudaStream_t s) {
    require(t>0 && t<=32 && checkpoint>=t && checkpoint%t==0,"Preview width must divide unchanged checkpoint block size");
    require(kd==128&&vd==128,"Unsupported GDN preview dimensions");preview_kernel<<<t*vh,128,0,s>>>(mixed,q,k,g,scale,bias,hc,state,y,t,kh,vh,kd,vd,checkpoint);
}
}
