#pragma once
#include "common.hpp"

namespace strata::qwen36_bf16::ops {
void cast(const float* in,BF* out,int count,cudaStream_t stream);
void residual(const BF* a,const BF* b,BF* out,int count,cudaStream_t stream);
void norm(const BF* in,const BF* weight,const BF* gate,BF* out,int rows,int width,float eps,bool zero_centered,cudaStream_t stream);
void swiglu(const BF* gate,const BF* up,BF* out,int count,cudaStream_t stream);
void sigmoid_mul(const BF* in,const BF* gate,BF* out,int rows,int width,bool scalar_gate,cudaStream_t stream);
void embedding(const BF* table,const int* ids,BF* out,int tokens,int width,cudaStream_t stream);
void prepare_q(const BF* projection,const BF* weight,BF* query,BF* gate,int tokens,int heads,int dim,int rotary_dim,int prefix,float theta,float eps,cudaStream_t stream);
void prepare_k(const BF* projection,const BF* weight,BF* key,int tokens,int heads,int dim,int rotary_dim,int prefix,float theta,float eps,cudaStream_t stream);
void attention(const BF* query,const BF* new_key,const BF* new_value,const BF* key_cache,const BF* value_cache,BF* out,int tokens,int prefix,int heads,int kv_heads,int dim,bool preview,cudaStream_t stream);
void gdn_conv(const BF* raw,const BF* history,const BF* left,const BF* right,BF* out,int tokens,int channels,int taps,bool preview,cudaStream_t stream);
void gdn_commit_conv(const BF* raw,BF* history,int tokens,int channels,int taps,cudaStream_t stream);
void gdn_qk(const BF* mixed,float* q,float* k,int tokens,int kh,int vh,int kd,int vd,cudaStream_t stream);
void gdn_factors(const BF* a,const BF* b,const BF* A_log,const BF* dt_bias,float* g,BF* beta,int tokens,int heads,cudaStream_t stream);
// A null output state guarantees read-only verification. Persistent state has
// layout [VH,VD,KD], matching the HF reference. KD=VD=128 in this first kernel.
void gdn_scan(const BF* mixed,const float* q,const float* k,const float* g,const BF* beta,const float* state_in,float* state_out,BF* out,int tokens,int kh,int vh,int kd,int vd,cudaStream_t stream);
void gdn_preview(const BF* mixed,const float* q,const float* k,const float* g,const BF* scale,const BF* bias,const BF* history_c,const float* state,BF* out,int tokens,int kh,int vh,int kd,int vd,int checkpoint_block,cudaStream_t stream);
}
