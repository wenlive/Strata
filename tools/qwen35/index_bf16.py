"""Stream-verify the complete official checkpoint; index all 40 text layers.

Vision and MTP are explicitly classified, retained on disk and excluded from
this causal text executor. A metadata audit is never executable.
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
    return {k:v for k,v in s.items() if not any(x in k for x in ("conv_right_weight","bd_softmax"))}

def validate(config,headers,weight_map):
    expected=expected_shapes(config);tensors={};excluded={};seen=set();totals=collections.Counter()
    for filename,file in headers.items():
        intervals=[]
        for name,t in file['header'].items():
            if name=='__metadata__':continue
            if name in seen:raise ValueError(f'Duplicate tensor: {name}')
            seen.add(name)
            if weight_map.get(name)!=filename:raise ValueError(f'Official index/header mismatch: {name}')
            if t['dtype']!='BF16':raise ValueError(f'Non-BF16 tensor: {name} {t["dtype"]}')
            shape=t['shape'];start,end=t['data_offsets'];size=math.prod(shape)*2
            if not shape or any(x<=0 for x in shape) or not (0<=start<=end and end-start==size and end+file['data_start']<=file['file_size']):raise ValueError(f'Invalid shape/range: {name}')
            intervals.append((start,end))
            desc={'source_tensor':name,'file':filename,'offset':start+file['data_start'],'bytes':size,'shape':shape,'dtype':'BF16','layout':'row-major'}
            if name.startswith('model.visual.') or name.startswith('mtp.'):
                group='vision_not_executed' if name.startswith('model.visual.') else 'mtp_not_executed';excluded[name]=desc
            else:
                alias=name.replace('model.language_model.','model.',1)
                if alias not in expected or shape!=expected[alias]:raise ValueError(f'Wrong text shape: {name}: {shape}; expected {expected.get(alias)}')
                if alias in tensors:raise ValueError(f'Alias collision: {alias}')
                tensors[alias]=desc
                group=('routed_experts' if '.experts.' in alias else 'shared_expert_and_gate' if '.shared_expert' in alias else 'embedding_and_lm_head' if alias in ('model.embed_tokens.weight','lm_head.weight') else 'gdn_base' if '.linear_attn.' in alias else 'full_attention' if '.self_attn.' in alias else 'norm_and_router')
            totals[group]+=size
        intervals.sort()
        if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError(f'Overlapping tensors: {filename}')
    if seen!=set(weight_map):raise ValueError(f'Checkpoint coverage mismatch: missing={set(weight_map)-seen}, extra={seen-set(weight_map)}')
    if set(tensors)!=set(expected):raise ValueError(f'Missing text tensors: {set(expected)-set(tensors)}')
    experts=[];h=config['hidden_size'];ff=config['moe_intermediate_size'];gu_size=2*ff*h*2;down_size=h*ff*2
    for layer in range(config['num_hidden_layers']):
        gu=tensors[f'model.layers.{layer}.mlp.experts.gate_up_proj'];down=tensors[f'model.layers.{layer}.mlp.experts.down_proj']
        for expert in range(config['num_experts']):
            experts.append({'layer':layer,'expert':expert,'gate_up':{'file':gu['file'],'offset':gu['offset']+expert*gu_size,'bytes':gu_size},'down':{'file':down['file'],'offset':down['offset']+expert*down_size,'bytes':down_size}})
    return tensors,experts,excluded,dict(totals)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-headers',type=Path,help='Remote-header audit only: complete=false, cannot execute')
    a=p.parse_args();source=json.loads((a.model/'source-manifest.json').read_text());api={x['rfilename']:x for x in source['siblings']}
    if source['sha']!='995ad96eacd98c81ed38be0c5b274b04031597b0':raise ValueError('Unvalidated checkpoint revision')
    raw_config=json.loads((a.model/'config.json').read_text());c=dict(raw_config['text_config']);c['num_shared_experts']=1
    if raw_config['model_type']!='qwen3_5_moe' or c['num_hidden_layers']!=40:raise ValueError('Unsupported checkpoint architecture')
    wm=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    controls=['config.json','generation_config.json','model.safetensors.index.json']
    if not a.audit_headers:controls += [x for x in ('tokenizer_config.json','tokenizer.json','chat_template.jinja','vocab.json','merges.txt') if x in api]
    for name in controls:
        path=a.model/name;info=api[name]
        if path.stat().st_size!=info['size']:raise ValueError(f'Control file size mismatch: {name}')
        sha256=hashlib.sha256();git=hashlib.sha1(b'blob '+str(info['size']).encode()+b'\0')
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):sha256.update(block);git.update(block)
        lfs=info.get('lfs')
        if (lfs and sha256.hexdigest()!=lfs['sha256']) or (not lfs and info.get('blobId') and git.hexdigest()!=info['blobId']):raise ValueError(f'Official provenance mismatch: {name}')
    if a.audit_headers:headers=json.loads(a.audit_headers.read_text())
    else:
        headers={}
        for filename in sorted(set(wm.values())):
            path=a.model/filename;size=path.stat().st_size
            if size!=api[filename]['size']:raise ValueError(f'Missing/truncated source: {filename}')
            digest=hashlib.sha256()
            with path.open('rb') as f:
                length=struct.unpack('<Q',f.read(8))[0]
                if not 2<=length<=16*1024**2:raise ValueError(f'Invalid header length: {filename}')
                header=json.loads(f.read(length));f.seek(0)
                for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
            if digest.hexdigest()!=api[filename]['lfs']['sha256']:raise ValueError(f'Checkpoint SHA256 mismatch: {filename}')
            headers[filename]={'file_size':size,'data_start':8+length,'header':header,'sha256':digest.hexdigest()};print('verified',filename,flush=True)
    tensors,experts,excluded,totals=validate(c,headers,wm)
    result={'format':'strata-qwen36-bf16-index','version':1,'complete':not bool(a.audit_headers),'scope':'metadata-audit' if a.audit_headers else 'full-text-model','repository':'Qwen/Qwen3.6-35B-A3B','revision':source['sha'],'model_root':str(a.model.resolve()),'config':c,'source_config':raw_config,'tensors':tensors,'experts':experts,'excluded_tensors':excluded,'capacity_bytes':totals,'total_checkpoint_weight_bytes':sum(totals.values()),'total_text_weight_bytes':sum(t['bytes'] for t in tensors.values()),'files':{name:{k:v for k,v in h.items() if k!='header'} for name,h in headers.items()}}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.name+'.incomplete')
    with tmp.open('w',encoding='utf-8') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,a.output);print(json.dumps({'complete':result['complete'],'capacity_bytes':totals,'text_weight_bytes':result['total_text_weight_bytes'],'experts':len(experts)},indent=2))


if __name__=='__main__':main()
