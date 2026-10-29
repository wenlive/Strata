"""Selected real Qwen decoder layers, streaming experts; not full-model validation.

The native test scope omits the final model norm/head. All layer and embedding
weights come from complete SHA256-verified official shards, never fake tensors.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def read_tensor(root,desc,shape=None):
    with (root/desc['file']).open('rb') as f:
        f.seek(desc['offset']);raw=bytearray(f.read(desc['bytes']))
    assert len(raw)==desc['bytes']
    return torch.frombuffer(raw,dtype=torch.bfloat16).clone().reshape(shape or desc['shape'])


def make_index(model,layer,output):
    raw=json.loads((model/'config.json').read_text());c=dict(raw['text_config']);c['num_shared_experts']=1
    wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    source=json.loads((model/'source-manifest.json').read_text());api={x['rfilename']:x for x in source['siblings']}
    prefix=f'model.language_model.layers.{layer}.'
    names=[n for n in wm if n.startswith(prefix) or n=='model.language_model.embed_tokens.weight']
    tensors={};files={}
    for filename in sorted({wm[n] for n in names}):
        path=model/filename;assert path.stat().st_size==api[filename]['size'],f'Missing complete shard {filename}'
        digest=hashlib.sha256()
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));f.seek(0)
            for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
        assert digest.hexdigest()==api[filename]['lfs']['sha256']
        files[filename]={'file_size':path.stat().st_size,'sha256':digest.hexdigest()}
        for name in names:
            if name not in header:continue
            t=header[name];assert t['dtype']=='BF16';lo,hi=t['data_offsets']
            alias=name.replace(prefix,'model.layers.0.').replace('model.language_model.','model.')
            tensors[alias]={'source_tensor':name,'file':filename,'offset':8+length+lo,'bytes':hi-lo,'shape':t['shape'],'dtype':'BF16','layout':'row-major'}
    gu=tensors['model.layers.0.mlp.experts.gate_up_proj'];down=tensors['model.layers.0.mlp.experts.down_proj'];experts=[]
    for e in range(256):experts.append({'layer':0,'expert':e,'gate_up':{'file':gu['file'],'offset':gu['offset']+e*4194304,'bytes':4194304},'down':{'file':down['file'],'offset':down['offset']+e*2097152,'bytes':2097152}})
    c['num_hidden_layers']=1;c['layer_types']=[c['layer_types'][layer]]
    result={'format':'strata-qwen36-bf16-index','version':1,'complete':True,'scope':'selected-official-layer-test','model_root':str(model.resolve()),'revision':source['sha'],'config':c,'selected_source_layer':layer,'tensors':tensors,'experts':experts,'files':files}
    output.write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',type=int,nargs='+',default=[0,3])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from transformers import AutoTokenizer,DynamicCache
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    text=tokenizer.apply_chat_template([{'role':'user','content':'Say hello briefly.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    base=tokenizer(text,add_special_tokens=False)['input_ids']*4
    report={'kind':'selected-official-Qwen3.6-layers','torch':torch.__version__,'reference_source':hf.__file__,'reference_sha256':hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest(),'attention_backend':'math SDPA','passed':True,'layers':[],'failures':[]}
    for layer_id in a.layers:
        root=a.output/f'layer{layer_id}';root.mkdir();index=make_index(a.model,layer_id,root/'index.json')
        steps=[{'tokens':base[:5],'preview':False,'commit':True},
               {'tokens':base[5:13],'preview':False,'commit':False},
               {'tokens':base[5:6],'preview':False,'commit':True},
               {'tokens':base[6:14],'preview':False,'commit':True},
               {'tokens':base[14:30],'preview':False,'commit':True},
               {'tokens':base[30:62],'preview':False,'commit':False}]
        job=root/'job.json';job.write_text(json.dumps({'index':str(root/'index.json'),'steps':steps},indent=2))
        with (root/'native.log').open('w') as log:rc=subprocess.run([str(a.exe),str(job),str(root/'native')],stdout=log,stderr=subprocess.STDOUT)
        assert rc.returncode==0,(root/'native.log').read_text()[-3000:]
        c=Qwen3_5MoeTextConfig(**index['config']);c._attn_implementation='sdpa'
        with torch.device('meta'):layer=hf.Qwen3_5MoeDecoderLayer(c,0)
        for name,param in list(layer.named_parameters()):
            if name.startswith('mlp.experts.'):continue
            parent,leaf=name.rsplit('.',1);value=read_tensor(a.model,index['tensors']['model.layers.0.'+name]).cuda()
            layer.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(value,requires_grad=False))
        traces={};routing={}

        class StreamingExperts(torch.nn.Module):
            def forward(self,x,ids,weights):
                out=torch.zeros_like(x)
                for e in ids.unique(sorted=True).tolist():
                    token,slot=torch.where(ids==e);entry=index['experts'][e]
                    gu=read_tensor(a.model,entry['gate_up'],(1024,2048)).cuda();down=read_tensor(a.model,entry['down'],(2048,512)).cuda()
                    gate,up=F.linear(x[token],gu).chunk(2,-1);y=F.linear(F.silu(gate)*up,down)*weights[token,slot,None]
                    out.index_add_(0,token,y.to(out.dtype))
                return out

        layer.mlp.experts=StreamingExperts();layer.eval();hooks=[]
        for key,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:hooks.append(module.register_forward_hook(lambda m,inp,out,key=key:traces.__setitem__(key,out.detach().clone())))
        def router_hook(m,inp,out):
            traces['router']=out[0].detach().clone();routing['ids']=out[2].detach().cpu().flatten();routing['weights']=out[1].detach().float().cpu().flatten()
        hooks.append(layer.mlp.gate.register_forward_hook(router_hook))
        mixer=layer.linear_attn if c.layer_types[0]=='linear_attention' else layer.self_attn
        hooks.append(mixer.register_forward_hook(lambda m,inp,out:traces.__setitem__('mixer',(out[0] if isinstance(out,tuple) else out).detach().clone())))
        if c.layer_types[0]=='linear_attention':
            hooks.append(mixer.in_proj_qkv.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_raw',out.detach().clone())))
            hooks.append(mixer.norm.register_forward_pre_hook(lambda m,inp:traces.__setitem__('gdn_y',inp[0].detach().clone())))
            hooks.append(mixer.norm.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_normalized',out.detach().clone())))
        cache=DynamicCache(config=c);prefix=0;rows=[];routes=[]

        def compare(name,ref,path):
            raw=path.read_bytes();got=torch.from_numpy(np.frombuffer(raw,dtype=np.float32 if path.suffix=='.f32' else np.uint16).copy())
            if path.suffix!='.f32':got=got.view(torch.bfloat16)
            ref=ref.detach().float().cpu().flatten();got=got.float().flatten();assert ref.shape==got.shape,(name,ref.shape,got.shape)
            delta=(ref-got).abs();maximum=delta.max().item() if len(delta) else 0;mean=delta.mean().item() if len(delta) else 0;rms=((ref-got).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item() if len(delta) else 0
            passed=rms<0.004 and maximum<=0.02*max(ref.abs().max().item() if len(ref) else 0,1e-6)+1e-6
            row={'tensor':name,'max_abs_error':maximum,'mean_abs_error':mean,'relative_rms_error':rms,'different_components':int((ref!=got).sum()),'passed':passed};rows.append(row)
            if not passed:report['failures'].append({'layer':layer_id,**row})

        rotary=hf.Qwen3_5MoeTextRotaryEmbedding(c).cuda()
        with torch.inference_mode(),torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            for step_id,step in enumerate(steps):
                t=len(step['tokens']);emb=index['tensors']['model.embed_tokens.weight'];xs=[]
                for token in step['tokens']:xs.append(read_tensor(a.model,{'file':emb['file'],'offset':emb['offset']+token*4096,'bytes':4096},(2048,)))
                x=torch.stack(xs)[None].cuda();positions=torch.arange(prefix,prefix+t,device='cuda')[None].expand(3,1,t)
                mask=None
                if c.layer_types[0]=='full_attention':mask=(torch.arange(prefix+t,device='cuda')[None,:]<=prefix+torch.arange(t,device='cuda')[:,None])[None,None]
                working=cache if step['commit'] else copy.deepcopy(cache);traces.clear();routing.clear()
                y=layer(x,position_embeddings=rotary(x,positions),attention_mask=mask,past_key_values=working)
                traces['output']=y;trace=root/'native'/f'step{step_id}'/'trace'
                for name,value in traces.items():compare(f'step{step_id}/{name}',value,trace/f'layer0-{name}.bf16')
                nr=json.loads((trace/'layer0-routes.json').read_text());got_ids=torch.tensor(nr['ids']);different=int((got_ids!=routing['ids']).sum())
                # Torch topk does not promise a tie order. Expert execution is
                # grouped by ID, so compare the expert->weight mapping as well
                # as recording every original slot difference. A changed set
                # remains a failure, even if the boundary scores tie.
                gi,go=got_ids.reshape(t,8).sort(-1);ri,ro=routing['ids'].reshape(t,8).sort(-1)
                gw=torch.tensor(nr['weights']).reshape(t,8).gather(-1,go);rw=routing['weights'].reshape(t,8).gather(-1,ro)
                set_rows=int((gi!=ri).any(-1).sum());weight_delta=(gw-rw).abs()
                weight_rms=float(weight_delta.square().mean().sqrt()/rw.square().mean().sqrt().clamp_min(1e-12))
                ulps=(gw.bfloat16().view(torch.int16).int()-rw.bfloat16().view(torch.int16).int()).abs()
                weights_pass=weight_rms<0.004 and float(weight_delta.max())<=0.02*max(float(rw.abs().max()),1e-6)+1e-6
                route={'step':step_id,'different_route_slots':different,'different_expert_set_rows':set_rows,'max_per_expert_weight_error':float(weight_delta.max()),'max_bf16_ulp_error':int(ulps.max()),'per_expert_weight_relative_rms':weight_rms,'weights_passed':weights_pass,'native_ids':got_ids.tolist(),'reference_ids':routing['ids'].tolist(),'native_weights':nr['weights'],'reference_weights':routing['weights'].tolist()};routes.append(route)
                if set_rows or not weights_pass:report['failures'].append({'layer':layer_id,**route})
                if step['commit']:prefix+=t
                state=root/'native'/f'step{step_id}'/'state'
                if prefix:
                    if c.layer_types[0]=='linear_attention':
                        compare(f'step{step_id}/conv_state',cache.get_conv_state(0,0)[...,-3:],state/'layer0-conv.bf16')
                        compare(f'step{step_id}/ssm_state',cache.get_recurrent_state(0,0).transpose(-1,-2),state/'layer0-ssm.f32')
                    else:
                        compare(f'step{step_id}/key_state',cache.layers[0].keys.transpose(1,2),state/'layer0-key.bf16')
                        compare(f'step{step_id}/value_state',cache.layers[0].values.transpose(1,2),state/'layer0-value.bf16')
        for hook in hooks:hook.remove()
        report['layers'].append({'source_layer':layer_id,'kind':c.layer_types[0],'comparisons':rows,'routing':routes})
        print(json.dumps({'layer':layer_id,'max_relative_rms':max(x['relative_rms_error'] for x in rows),'route_slots_different':sum(x['different_route_slots'] for x in routes)}),flush=True)
        del layer,cache,working,rotary;torch.cuda.empty_cache()
    report['passed']=not report['failures'];(a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':report['passed'],'failures':report['failures']},indent=2));assert report['passed']


if __name__=='__main__':main()
