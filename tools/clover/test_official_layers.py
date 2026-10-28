"""Selected actual Clover decoder-layer oracle with streaming expert reads.

The generation executable rejects these selected-layer indices. This harness
uses the same native Model implementation, without claiming full-model support.
"""
import argparse
import ast
import copy
import contextlib
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def read_tensor(root,desc,shape=None):
    with (root/desc['file']).open('rb') as f:
        f.seek(desc['offset']);raw=bytearray(f.read(desc['bytes']))
    assert len(raw)==desc['bytes']
    return torch.frombuffer(raw,dtype=torch.bfloat16).clone().reshape(shape or desc['shape'])


def make_index(model,layer,output):
    c=json.loads((model/'config.json').read_text())
    wm=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    source=json.loads((model/'source-manifest.json').read_text());api={x['rfilename']:x for x in source['siblings']}
    names=[n for n in wm if n.startswith(f'model.layers.{layer}.') or n in ('model.embed_tokens.weight','model.norm.weight','lm_head.weight')]
    tensors={};files={}
    for filename in sorted({wm[n] for n in names}):
        path=model/filename
        assert path.stat().st_size==api[filename]['size'],f'Missing complete source {filename}'
        h=hashlib.sha256()
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));f.seek(0)
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        assert h.hexdigest()==api[filename]['lfs']['sha256']
        files[filename]={'file_size':path.stat().st_size,'sha256':h.hexdigest()}
        for n in names:
            if n not in header:continue
            t=header[n];assert t['dtype']=='BF16';lo,hi=t['data_offsets']
            alias=n.replace(f'model.layers.{layer}.','model.layers.0.')
            tensors[alias]={'source_tensor':n,'file':filename,'offset':length+8+lo,'bytes':hi-lo,
                            'dtype':'BF16','layout':'row-major','shape':t['shape']}
    experts=[];gu=tensors['model.layers.0.mlp.experts.gate_up_proj'];down=tensors['model.layers.0.mlp.experts.down_proj']
    for e in range(256):
        experts.append({'layer':0,'expert':e,'gate_up':{'file':gu['file'],'offset':gu['offset']+e*4194304,'bytes':4194304},
                        'down':{'file':down['file'],'offset':down['offset']+e*2097152,'bytes':2097152}})
    test_c=copy.deepcopy(c);test_c['num_hidden_layers']=1;test_c['layer_types']=[c['layer_types'][layer]]
    index={'format':'strata-clover-bf16-index','version':1,'complete':True,'scope':'selected-official-layer-test',
           'model_root':str(model.resolve()),'revision':source['sha'],'config':test_c,'source_config':c,
           'selected_source_layer':layer,'tensors':tensors,'files':files,'experts':experts}
    output.write_text(json.dumps(index,indent=2));return index


def oracle_namespace(path):
    names={'bd_gdn_conv1d','bd_gdn_local_memory','recurrent_commit_torch','coreset_torch','linear_fp32_output',
           'causal_conv_silu','CloverRMSNorm','CloverAttention','CloverGatedDeltaNet','CloverMLP','CloverExperts',
           'CloverMoeBlock','CloverDecoderLayer'}
    code=path.read_text();nodes=[n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    ns={'torch':torch,'F':F,'nn':torch.nn};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--layers',type=int,nargs='+',default=[0,3])
    p.add_argument('--sdpa-backend',choices=('auto','math'),default='auto')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    ns=oracle_namespace(a.model/'modeling_clover_moe.py')
    report={'kind':'selected-official-decoder-layers','passed':True,'layers':[],'failures':[],'torch':torch.__version__,'sdpa_backend':a.sdpa_backend}
    source=json.loads((a.model/'source-manifest.json').read_text());report['revision']=source['sha']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.model),trust_remote_code=False)
    text=tok.apply_chat_template([{'role':'user','content':'Say hello briefly.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    base=tok(text,add_special_tokens=False)['input_ids']*4
    for layer_id in a.layers:
        root=a.output/f'layer{layer_id}';root.mkdir();ipath=root/'index.json';index=make_index(a.model,layer_id,ipath)
        mask=index['config']['mask_token_id']
        steps=[{'tokens':base[:5],'preview':False,'commit':True},
               {'tokens':[base[5]]+[mask]*7,'preview':True,'commit':False},
               {'tokens':base[5:13],'preview':False,'commit':False},
               {'tokens':base[5:8],'preview':False,'commit':True},
               {'tokens':[base[8]]+[mask]*15,'preview':True,'commit':False},
               {'tokens':base[8:24],'preview':False,'commit':False},
               {'tokens':base[8:24],'preview':False,'commit':True},
               {'tokens':[base[24]]+[mask]*31,'preview':True,'commit':False},
               {'tokens':base[24:56],'preview':False,'commit':False},
               {'tokens':[base[24]],'preview':False,'commit':True}]
        job=root/'job.json';job.write_text(json.dumps({'index':str(ipath),'steps':steps},indent=2))
        native=root/'native'
        with (root/'native.log').open('w') as log:
            result=subprocess.run([str(a.exe),str(job),str(native)],stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(root/'native.log').read_text()[-4000:]
        c=SimpleNamespace(**index['config'])
        # Instantiate only meta tensors, then materialize dense parameters.
        # The routed bank is replaced before any real memory is allocated.
        with torch.device('meta'):layer=ns['CloverDecoderLayer'](c,0)
        for name,param in list(layer.named_parameters()):
            if name.startswith('mlp.experts.'):continue
            desc=index['tensors']['model.layers.0.'+name]
            parent,leaf=name.rsplit('.',1)
            layer.get_submodule(parent).register_parameter(leaf,torch.nn.Parameter(read_tensor(a.model,desc).cuda(),requires_grad=False))
        traces={};routing={}

        class StreamingExperts(torch.nn.Module):
            def forward(self,x,ids,weights):
                routing['ids']=ids.detach().cpu().flatten();routing['weights']=weights.detach().cpu().flatten()
                out=torch.zeros_like(x,dtype=torch.float32)
                for e in ids.unique(sorted=True).tolist():
                    token,slot=torch.where(ids==e);entry=index['experts'][e]
                    gu=read_tensor(a.model,entry['gate_up'],(1024,2048)).cuda()
                    down=read_tensor(a.model,entry['down'],(2048,512)).cuda()
                    g,u=F.linear(x[token],gu).chunk(2,-1);hidden=(F.silu(g.float())*u.float()).bfloat16()
                    y=ns['linear_fp32_output'](hidden,down)
                    out.index_add_(0,token,(y*weights[token,slot,None].float()).bfloat16().float())
                return out.bfloat16()
        layer.mlp.experts=StreamingExperts();layer.eval()
        hooks=[]
        for key,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm),('router',layer.mlp.gate)]:
            hooks.append(module.register_forward_hook(lambda m,inp,out,key=key:traces.__setitem__(key,out.detach().clone())))
        mixer=layer.linear_attn if hasattr(layer,'linear_attn') else layer.self_attn
        hooks.append(mixer.register_forward_hook(lambda m,inp,out:traces.__setitem__('mixer',out[0].detach().clone())))
        if hasattr(layer,'linear_attn'):
            hooks.append(mixer.in_proj_qkv.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_raw',out.detach().clone())))
            hooks.append(mixer.norm.register_forward_pre_hook(lambda m,inp:traces.__setitem__('gdn_y',inp[0].detach().clone())))
            hooks.append(mixer.norm.register_forward_hook(lambda m,inp,out:traces.__setitem__('gdn_normalized',out.detach().clone())))
        cache=None;prefix=0;layer_report={'source_layer':layer_id,'kind':c.layer_types[0],'comparisons':[],'routing':[]}

        def compare(name,ref,filename,fp32=False):
            raw=filename.read_bytes();dtype=np.float32 if fp32 else np.uint16
            got=torch.from_numpy(np.frombuffer(raw,dtype=dtype).copy())
            if not fp32:got=got.view(torch.bfloat16)
            got=got.float().flatten();ref=ref.detach().float().cpu().flatten()
            assert got.shape==ref.shape,(name,got.shape,ref.shape)
            diff=(got-ref).abs();maximum=diff.max().item() if diff.numel() else 0
            mean=diff.mean().item() if diff.numel() else 0
            rms=((got-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item() if diff.numel() else 0
            passed=rms<0.004 and maximum<=0.02*max(ref.abs().max().item() if ref.numel() else 0,1e-6)+1e-6
            # This initial diagnostic bound is deliberately reported; it is not
            # an end-to-end model acceptance criterion. Routing is checked too.
            entry={'tensor':name,'max_abs_error':maximum,'mean_abs_error':mean,'relative_rms_error':rms,'different_components':int((got!=ref).sum()),'diagnostic_passed':passed}
            layer_report['comparisons'].append(entry)
            if not passed:report['failures'].append({'layer':layer_id,**entry})

        sdpa_context=(torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]) if a.sdpa_backend=='math' else contextlib.nullcontext())
        with torch.inference_mode(),sdpa_context:
            for step_id,step in enumerate(steps):
                ids=step['tokens'];t=len(ids);traces.clear();routing.clear()
                emb=index['tensors']['model.embed_tokens.weight'];xs=[]
                for token in ids:
                    xs.append(read_tensor(a.model,{'file':emb['file'],'offset':emb['offset']+token*4096,'bytes':4096},(2048,)))
                x=torch.stack(xs)[None].cuda()
                valid=torch.ones(1,t,device='cuda',dtype=torch.bool);prefix_valid=torch.ones(1,prefix,device='cuda',dtype=torch.bool)
                positions=prefix+torch.arange(t,device='cuda')[None]
                y,next_cache=layer(x,cache,valid,prefix_valid,positions,step['preview'])
                traces['output']=y
                if step['commit']:cache=next_cache;prefix+=t
                trace=native/f'step{step_id}'/'trace'
                compare(f'step{step_id}/embedding',x,trace/'embedding.bf16')
                for name,value in traces.items():compare(f'step{step_id}/{name}',value,trace/f'layer0-{name}.bf16')
                nr=json.loads((trace/'layer0-routes.json').read_text())
                got_ids=torch.tensor(nr['ids']);got_w=torch.tensor(nr['weights'])
                route_diffs=int((got_ids!=routing['ids']).sum());werr=(got_w-routing['weights']).abs().max().item()
                coreset_equal=True
                if step['preview']:
                    masked=ns['coreset_torch'](traces['router'],valid,8,0.9)
                    allowed=torch.isfinite(masked[0,0]).nonzero().flatten().cpu().tolist()
                    coreset_equal=allowed==sorted(nr['coreset'])
                layer_report['routing'].append({'step':step_id,'different_route_slots':route_diffs,'max_weight_error':werr,'coreset_equal':coreset_equal})
                if route_diffs:report['failures'].append({'layer':layer_id,'step':step_id,'different_route_slots':route_diffs})
                if not coreset_equal:report['failures'].append({'layer':layer_id,'step':step_id,'coreset_equal':False})
                state_dir=native/f'step{step_id}'/'state'
                assert json.loads((state_dir/'manifest.json').read_text())['prefix_tokens']==prefix
                if hasattr(layer,'linear_attn'):
                    compare(f'step{step_id}/conv_state',cache['conv'],state_dir/'layer0-conv.bf16')
                    compare(f'step{step_id}/ssm_state',cache['ssm'],state_dir/'layer0-ssm.f32',True)
                else:
                    compare(f'step{step_id}/key_state',cache['key'].transpose(1,2),state_dir/'layer0-key.bf16')
                    compare(f'step{step_id}/value_state',cache['value'].transpose(1,2),state_dir/'layer0-value.bf16')
        report['layers'].append(layer_report)
        print(json.dumps({'layer':layer_id,'comparisons':len(layer_report['comparisons']),'route_differences':sum(r['different_route_slots'] for r in layer_report['routing']),
                          'max_relative_rms':max(r['relative_rms_error'] for r in layer_report['comparisons'])}),flush=True)
        for hook in hooks:hook.remove()
        del layer;torch.cuda.empty_cache()
    report['passed']=not report['failures'];(a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'failures':report['failures']},indent=2),flush=True)
    assert report['passed']


if __name__=='__main__':main()
