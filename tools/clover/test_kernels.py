"""Compare native Clover attention/GDN/router primitives to pinned HF source.

Synthetic tensors only. The official standalone functions are extracted by
AST for the oracle; no model is loaded, and this is not an inference path.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def load_oracle(path):
    code=path.read_text(encoding='utf-8')
    names={'bd_gdn_conv1d','bd_gdn_local_memory','recurrent_commit_torch','coreset_torch',
           'causal_conv_silu','CloverRMSNorm','CloverAttention'}
    nodes=[x for x in ast.parse(code).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert {x.name for x in nodes}==names
    namespace={'torch':torch,'F':F,'nn':torch.nn}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--hf-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    oracle=load_oracle(a.hf_source)
    torch.manual_seed(821)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    report={'kind':'synthetic-kernels-only','torch':torch.__version__,
            'source':str(a.hf_source),'source_sha256':hashlib.sha256(a.hf_source.read_bytes()).hexdigest(),
            'comparisons':[],'failures':[]}
    refs={};fixture=a.output/'fixture';fixture.mkdir();job={}

    def write(name,tensor):
        tensor=tensor.detach().cpu().contiguous()
        raw=tensor.view(torch.uint8).numpy().tobytes()
        file=fixture/(name+'.bin');file.write_bytes(raw)
        return {'file':str(file),'bytes':len(raw)}

    def random(*shape,scale=0.3,dtype=torch.bfloat16):
        return (torch.randn(*shape)*scale).to(dtype).cuda()

    kh,vh,kd,vd,ch=16,32,128,128,8192
    tensors={'history':random(ch,3),'state':random(vh,vd,kd,scale=0.02,dtype=torch.float32),
             'left':random(ch,1,4),'right':random(ch,1,3),
             'scale':random(vh,scale=0.1)+10,'bias':random(vh,scale=0.1)+5,
             'history_c':random(vh,scale=0.1)+1,'A_log':random(vh,scale=0.2),
             'dt_bias':random(vh,scale=0.2),'norm':random(vd,scale=0.1)+1}
    gdn={'kh':kh,'vh':vh,'inputs':{n:write('gdn-'+n,t) for n,t in tensors.items()},'steps':[]}
    history=tensors['history'].clone();state=tensors['state'].clone()
    specs=[(5,False,True,None),(8,True,False,None),(8,False,False,None),(3,False,True,2),
           (16,True,False,None),(16,False,False,None),(16,False,True,5),
           (32,True,False,None),(32,False,False,None),(1,False,True,8),
           (8,True,False,None),(1,False,True,None)]
    saved=[]
    for n,(t,preview,commit,reuse) in enumerate(specs):
        inputs=({key:value[:t].clone() for key,value in saved[reuse].items()} if reuse is not None else
                {'raw':random(t,ch),'a':random(t,vh),'b':random(t,vh),'z':random(t,vh,vd)})
        saved.append(inputs)
        step={'tokens':t,'preview':preview,'commit':commit,
              **{key:write(f'gdn-step{n}-{key}',value) for key,value in inputs.items()}}
        gdn['steps'].append(step)
        raw=inputs['raw'];valid=torch.ones(1,t,device='cuda',dtype=torch.bool)
        g=-tensors['A_log'].float().exp()*F.softplus(inputs['a'].float()+tensors['dt_bias'].float())
        beta=inputs['b'].sigmoid()
        if preview:
            mixed=oracle['bd_gdn_conv1d'](raw,clean_conv_states=history[None],cache_indices=torch.tensor([0],device='cuda'),
                conv_weights=tensors['left'],block_size=t,active_valid_mask=valid.flatten(),conv_right_weight=tensors['right'])
        else:
            joined=torch.cat((history,raw.T),-1)[None]
            mixed=oracle['causal_conv_silu'](joined,tensors['left'])[0].T.contiguous()
        qb,kb,vb=mixed.split((kh*kd,kh*kd,vh*vd),-1)
        qb=qb.reshape(1,t,kh,kd);kb=kb.reshape(1,t,kh,kd);vb=vb.reshape(1,t,vh,vd)
        q=qb.float()*torch.rsqrt(qb.float().square().sum(-1,keepdim=True)+1e-6)
        k=kb.float()*torch.rsqrt(kb.float().square().sum(-1,keepdim=True)+1e-6)
        if preview:
            y=oracle['bd_gdn_local_memory'](qb,kb,vb,g[None],clean_ssm_states=state[None],
                cache_indices=torch.tensor([0],device='cuda'),active_valid_mask=valid.flatten(),scale=kd**-0.5,
                softmax_logit_scale=tensors['scale'],softmax_history_bias=tensors['bias'],softmax_history_c=tensors['history_c'])
        else:
            y,next_state=oracle['recurrent_commit_torch'](qb,kb,vb,g[None],beta[None],state[None],valid)
            if commit:
                state=next_state[0].clone();history=joined[0,...,-3:].clone()
        norm=oracle['CloverRMSNorm'](vd,1e-6,zero_centered=False).bfloat16().cuda()
        norm.weight.data.copy_(tensors['norm'])
        normalized=norm(y,inputs['z'][None])
        for name,value in {'mixed.bf16':mixed,'q.f32':q,'k.f32':k,'g.f32':g,'beta.bf16':beta,
                           'y.bf16':y,'normalized.bf16':normalized,'state.f32':state,'history.bf16':history}.items():
            refs[f'gdn-{n}/{name}']=value.detach().float().cpu().contiguous().flatten()
    job['gdn']=gdn

    job['attention']=[]
    for n,(t,prefix) in enumerate(((1,0),(8,3),(16,7),(32,5))):
        nh,nkv,dim=16,2,256
        inputs={'q_proj':random(t,nh,dim*2),'k_proj':random(t,nkv,dim),'v':random(t,nkv,dim),
                'q_norm':random(dim,scale=0.1),'k_norm':random(dim,scale=0.1),
                'prefix_key':random(prefix,nkv,dim),'prefix_value':random(prefix,nkv,dim)}
        job['attention'].append({'tokens':t,'prefix':prefix,'heads':nh,'kv_heads':nkv,
                                 'inputs':{key:write(f'attn{n}-{key}',value) for key,value in inputs.items()}})
        qp,gate=inputs['q_proj'].chunk(2,-1)
        qnorm=oracle['CloverRMSNorm'](dim,1e-6).bfloat16().cuda();qnorm.weight.data.copy_(inputs['q_norm'])
        knorm=oracle['CloverRMSNorm'](dim,1e-6).bfloat16().cuda();knorm.weight.data.copy_(inputs['k_norm'])
        positions=(prefix+torch.arange(t,device='cuda'))[None]
        dummy=SimpleNamespace(config=SimpleNamespace(head_dim=dim,partial_rotary_factor=0.25,rope_parameters={'rope_theta':10000000}))
        q=oracle['CloverAttention'].rotary(dummy,qnorm(qp[None]).transpose(1,2),positions)
        k=oracle['CloverAttention'].rotary(dummy,knorm(inputs['k_proj'][None]).transpose(1,2),positions)
        keys=torch.cat((inputs['prefix_key'][None].transpose(1,2),k),2).repeat_interleave(nh//nkv,1)
        values=torch.cat((inputs['prefix_value'][None].transpose(1,2),inputs['v'][None].transpose(1,2)),2).repeat_interleave(nh//nkv,1)
        for preview in (False,True):
            allow=torch.ones(t,prefix+t,device='cuda',dtype=torch.bool)
            if not preview:allow=torch.arange(prefix+t,device='cuda')[None]<=prefix+torch.arange(t,device='cuda')[:,None]
            y=F.scaled_dot_product_attention(q,keys,values,attn_mask=allow[None,None],dropout_p=0.0)[0].transpose(0,1)
            mode='preview' if preview else 'causal'
            refs[f'attention-{n}/{mode}.bf16']=y.float().cpu().flatten()
            refs[f'attention-{n}/{mode}_gated.bf16']=(y*gate.sigmoid()).float().cpu().flatten()
        refs[f'attention-{n}/q.bf16']=q[0].transpose(0,1).float().cpu().flatten()
        refs[f'attention-{n}/k.bf16']=k[0].transpose(0,1).float().cpu().flatten()
        refs[f'attention-{n}/gate.bf16']=gate.float().cpu().contiguous().flatten()

    job['router']=[];router_refs=[]
    for t in (1,8,32):
        for kind in ('random','all-ties','dominant'):
            logits=random(t,256,scale=1).float()
            if kind=='all-ties':logits.zero_()
            elif kind=='dominant':logits.fill_(-20);logits[:,17]=20
            for preview in (False,True):
                job['router'].append({'tokens':t,'experts':256,'top_k':8,'preview':preview,'top_p':0.9,
                                      'logits':logits.flatten().tolist()})
                masked=oracle['coreset_torch'](logits[None],torch.ones(1,t,device='cuda',dtype=torch.bool),8,0.9)[0] if preview else logits
                ids=masked.argsort(dim=-1,descending=True,stable=True)[:,:8]
                w=masked.softmax(-1).gather(-1,ids);w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
                router_refs.append((ids.cpu(),w.cpu(),torch.isfinite(masked[0]).nonzero().flatten().cpu() if preview else None))
    jobpath=a.output/'job.json';jobpath.write_text(json.dumps(job))
    native=a.output/'native'
    r=subprocess.run([str(a.exe),str(jobpath),str(native)],capture_output=True,text=True)
    (a.output/'stdout.txt').write_text(r.stdout);(a.output/'stderr.txt').write_text(r.stderr)
    assert r.returncode==0,r.stderr
    for name,ref in refs.items():
        raw=(native/name).read_bytes()
        got=torch.from_numpy(np.frombuffer(raw,dtype=np.uint16 if name.endswith('.bf16') else np.float32).copy())
        if name.endswith('.bf16'):got=got.view(torch.bfloat16)
        got=got.float().flatten();diff=(got-ref).abs()
        maximum=diff.max().item() if diff.numel() else 0
        mean=diff.mean().item() if diff.numel() else 0
        rms=((got-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item()
        fp32=name.endswith('.f32')
        passed=(rms<2e-5 and maximum<1e-5*max(1,ref.abs().max().item())) if fp32 else (rms<0.004 and maximum<=0.02*ref.abs().max().item()+1e-6)
        if name.endswith('history.bf16') or name.endswith('gate.bf16'):passed=torch.equal(got,ref)
        row={'tensor':name,'max_abs_error':maximum,'mean_abs_error':mean,'relative_rms_error':rms,
             'different_components':int((got!=ref).sum()),'passed':passed}
        report['comparisons'].append(row)
        if not passed:report['failures'].append(row)
    parsed=json.loads((native/'result.json').read_text())
    for n,(result,ref) in enumerate(zip(parsed['router'],router_refs)):
        ids,w,coreset=ref;got_ids=torch.tensor(result['ids']).reshape_as(ids);got_w=torch.tensor(result['weights']).reshape_as(w)
        err=(got_w-w).abs().max().item()
        good=torch.equal(ids,got_ids) and err<2e-6
        if coreset is not None:good=good and sorted(result['coreset'])==coreset.tolist()
        row={'router_case':n,'ids_equal':torch.equal(ids,got_ids),'max_weight_error':err,'passed':good}
        report['comparisons'].append(row)
        if not good:report['failures'].append(row)
    report['passed']=not report['failures']
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'comparisons':len(report['comparisons']),'failures':report['failures']},indent=2),flush=True)
    assert report['passed']


if __name__=='__main__':
    main()
