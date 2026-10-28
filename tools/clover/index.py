"""Validate official BF16 safetensors and atomically index tensor/expert slices.

No tensor payload is ever loaded as a whole. The optional metadata audit does
not produce a runtime index and does not assert that weights are downloaded.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import struct


def expected_shapes(c):
    d, ff, ne = c['hidden_size'], c['moe_intermediate_size'], c['num_experts']
    vh, kh = c['linear_num_value_heads'], c['linear_num_key_heads']
    vd, kd = c['linear_value_head_dim'], c['linear_key_head_dim']
    channels=2*kh*kd+vh*vd
    s={'model.embed_tokens.weight':[c['vocab_size'],d],
       'model.norm.weight':[d], 'lm_head.weight':[c['vocab_size'],d]}
    for i, kind in enumerate(c['layer_types']):
        pre=f'model.layers.{i}.'
        shapes={'input_layernorm.weight':[d], 'post_attention_layernorm.weight':[d],
                'mlp.gate.weight':[ne,d], 'mlp.experts.gate_up_proj':[ne,2*ff,d],
                'mlp.experts.down_proj':[ne,d,ff]}
        if c['num_shared_experts']!=1:
            raise ValueError('This adapter requires exactly one shared expert')
        sf=c['shared_expert_intermediate_size']
        shapes.update({'mlp.shared_expert.gate_proj.weight':[sf,d],
                       'mlp.shared_expert.up_proj.weight':[sf,d],
                       'mlp.shared_expert.down_proj.weight':[d,sf],
                       'mlp.shared_expert_gate.weight':[1,d]})
        if kind=='linear_attention':
            shapes.update({f'linear_attn.{n}':sh for n,sh in {
                'in_proj_qkv.weight':[channels,d], 'in_proj_z.weight':[vh*vd,d],
                'in_proj_a.weight':[vh,d], 'in_proj_b.weight':[vh,d],
                'out_proj.weight':[d,vh*vd], 'conv1d.weight':[channels,1,c['linear_conv_kernel_dim']],
                'conv_right_weight':[channels,1,c['linear_conv_kernel_dim']-1],
                'A_log':[vh], 'dt_bias':[vh], 'norm.weight':[vd],
                'bd_softmax_logit_scale':[vh], 'bd_softmax_history_bias':[vh],
                'bd_softmax_history_c':[vh]}.items()})
        elif kind=='full_attention':
            hd,nh,nkv=c['head_dim'],c['num_attention_heads'],c['num_key_value_heads']
            shapes.update({f'self_attn.{n}':sh for n,sh in {
                'q_proj.weight':[2*nh*hd,d], 'k_proj.weight':[nkv*hd,d],
                'v_proj.weight':[nkv*hd,d], 'o_proj.weight':[d,nh*hd],
                'q_norm.weight':[hd], 'k_norm.weight':[hd]}.items()})
        else:
            raise ValueError(f'Unknown layer type {kind}')
        s.update({pre+n:sh for n,sh in shapes.items()})
    return s


def validate(c, headers, weight_map):
    expected=expected_shapes(c)
    tensors={}
    totals=collections.Counter()
    for filename, file in headers.items():
        intervals=[]
        for name, t in file['header'].items():
            if name=='__metadata__': continue
            if name in tensors:
                raise ValueError(f'Duplicate tensor {name}')
            if name not in expected or t['shape']!=expected[name]:
                raise ValueError(f'Wrong or unexpected shape: {name}: {t["shape"]}; expected {expected.get(name)}')
            if t['dtype']!='BF16':
                raise ValueError(f'Wrong dtype: {name}: {t["dtype"]}')
            start,end=t['data_offsets']
            size=math.prod(t['shape'])*2
            if not (0<=start<=end and end-start==size and end+file['data_start']<=file['file_size']):
                raise ValueError(f'Invalid or truncated range {filename}:{name}')
            if weight_map.get(name)!=filename:
                raise ValueError(f'Official index/header mismatch: {name}')
            intervals.append((start,end))
            tensors[name]={'file':filename,'offset':start+file['data_start'],
                           'bytes':size,'shape':t['shape'],'dtype':'BF16','layout':'row-major'}
            group=('routed_experts' if '.experts.' in name else
                   'shared_expert_and_gate' if '.shared_expert' in name else
                   'clover_gdn_extra' if any(x in name for x in ('bd_softmax','conv_right')) else
                   'embedding_and_lm_head' if name in ('model.embed_tokens.weight','lm_head.weight') else
                   'gdn_base' if '.linear_attn.' in name else
                   'full_attention' if '.self_attn.' in name else 'norm_and_router')
            totals[group]+=size
        intervals.sort()
        if any(intervals[i][1]>intervals[i+1][0] for i in range(len(intervals)-1)):
            raise ValueError(f'Overlapping tensors in {filename}')
    if set(tensors)!=set(expected) or set(tensors)!=set(weight_map):
        raise ValueError(f'Missing tensors: {set(expected)-set(tensors)}; unexpected index keys: {set(weight_map)-set(expected)}')
    experts=[]
    d,ff=c['hidden_size'],c['moe_intermediate_size']
    gu_bytes=2*ff*d*2
    down_bytes=d*ff*2
    for l in range(c['num_hidden_layers']):
        gu=tensors[f'model.layers.{l}.mlp.experts.gate_up_proj']
        down=tensors[f'model.layers.{l}.mlp.experts.down_proj']
        for e in range(c['num_experts']):
            # gate then up are contiguous within [E,2*FF,D]. Down is [E,D,FF].
            experts.append({'layer':l,'expert':e,
                'gate_up':{'file':gu['file'],'offset':gu['offset']+e*gu_bytes,'bytes':gu_bytes},
                'down':{'file':down['file'],'offset':down['offset']+e*down_bytes,'bytes':down_bytes}})
    return tensors,experts,dict(totals)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-headers',type=Path,help='Metadata-only audit: emits no usable runtime index')
    a=p.parse_args()
    c=json.loads((a.model/'config.json').read_text(encoding='utf-8'))
    source=json.loads((a.model/'source-manifest.json').read_text(encoding='utf-8'))
    wm=json.loads((a.model/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    if a.audit_headers:
        headers=json.loads(a.audit_headers.read_text(encoding='utf-8'))
    else:
        headers={}
        api={x['rfilename']:x for x in source['siblings']}
        for file in sorted(set(wm.values())):
            path=a.model/file
            size=path.stat().st_size
            if size!=api[file]['size']:
                raise ValueError(f'Truncated/wrong file size: {file}: {size}')
            digest=hashlib.sha256()
            with path.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0]
                if not 2<=length<=16*1024**2:
                    raise ValueError(f'Invalid header length: {file}: {length}')
                raw=f.read(length)
                header=json.loads(raw)
                f.seek(0)
                for block in iter(lambda:f.read(1024**2),b''):
                    digest.update(block)
            sha=digest.hexdigest()
            if sha!=api[file]['lfs']['sha256']:
                raise ValueError(f'SHA256 mismatch: {file}')
            headers[file]={'file_size':size,'data_start':8+length,'header':header,'sha256':sha}
            print(f'verified {file}',flush=True)
    tensors,experts,totals=validate(c,headers,wm)
    result={'format':'strata-clover-bf16-index','version':1,
            'complete':not bool(a.audit_headers),'scope':'metadata-audit' if a.audit_headers else 'full-model',
            'repository':'Diffuspace/Clover-35B-A3B-Preview','revision':source['sha'],
            'model_root':str(a.model.resolve()),'config':c,'tensors':tensors,
            'experts':experts,'capacity_bytes':totals,
            'total_weight_bytes':sum(totals.values()),
            'files':{f:{k:v for k,v in h.items() if k!='header'} for f,h in headers.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_name(a.output.name+'.incomplete')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,a.output)
    print(json.dumps({'complete':result['complete'],'capacity_bytes':totals,
                     'total_weight_bytes':sum(totals.values()),'expert_bytes':3*c['hidden_size']*c['moe_intermediate_size']*2},indent=2))


if __name__=='__main__':
    main()
