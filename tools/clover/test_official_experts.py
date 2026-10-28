"""Selected official layer-0 BF16 expert oracle; never loads the entire checkpoint.

This validates real weights on synthetic BF16 activations. It is not a full
Clover layer or model test. Only two completed, SHA256-verified shards are needed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.model/'source-manifest.json').read_text())
    api={x['rfilename']:x for x in source['siblings']}
    wm=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    names=[f'model.layers.0.mlp.experts.{n}' for n in ('gate_up_proj','down_proj')]
    router='model.layers.0.mlp.gate.weight'
    files={};tensors={}
    for filename in sorted({wm[n] for n in names+[router]}):
        path=a.model/filename;assert path.stat().st_size==api[filename]['size']
        h=hashlib.sha256()
        with path.open('rb') as f:
            length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));f.seek(0)
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        assert h.hexdigest()==api[filename]['lfs']['sha256']
        files[filename]={'file_size':path.stat().st_size,'sha256':h.hexdigest()}
        for name in names+[router]:
            if name not in header:continue
            t=header[name];start,end=t['data_offsets'];assert t['dtype']=='BF16'
            tensors[name]={'file':filename,'offset':8+length+start,'bytes':end-start,'dtype':'BF16',
                           'shape':t['shape'],'layout':'row-major'}
    c=json.loads((a.model/'config.json').read_text());test_c=copy.deepcopy(c);test_c['num_hidden_layers']=1
    experts=[]
    for e in range(256):
        gu=tensors[names[0]];down=tensors[names[1]]
        experts.append({'layer':0,'expert':e,
            'gate_up':{'file':gu['file'],'offset':gu['offset']+e*4*1024**2,'bytes':4*1024**2},
            'down':{'file':down['file'],'offset':down['offset']+e*2*1024**2,'bytes':2*1024**2}})
    index={'format':'strata-clover-bf16-index','version':1,'scope':'selected-official-expert-test','complete':True,
           'model_root':str(a.model.resolve()),'revision':source['sha'],'config':test_c,'source_config':c,
           'selected_source_layer':0,'files':files,'tensors':tensors,'experts':experts}
    ipath=a.output/'selected-expert-index.json';ipath.write_text(json.dumps(index,indent=2))

    def read(slice,shape):
        with (a.model/slice['file']).open('rb') as f:
            f.seek(slice['offset']);data=bytearray(f.read(slice['bytes']))
        assert len(data)==slice['bytes']
        return torch.frombuffer(data,dtype=torch.bfloat16).clone().reshape(shape)

    report={'kind':'selected-official-experts-on-synthetic-activations','revision':source['sha'],
            'source_layer':0,'torch':torch.__version__,'cases':[],'failures':[]}
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    gate=read(tensors[router],(256,2048)).cuda()
    for rows in (1,8,32):
        torch.manual_seed(rows+522);x=torch.randn(rows,2048,device='cuda').bfloat16()
        with torch.inference_mode():
            logits=F.linear(x,gate).float();ids=logits.argsort(dim=-1,descending=True,stable=True)[:,:8]
            w=logits.softmax(-1).gather(-1,ids);w=w/w.sum(-1,keepdim=True)
            expected=torch.zeros(rows,2048,device='cuda',dtype=torch.float32)
            for e in ids.unique(sorted=True).tolist():
                token,slot=torch.where(ids==e);entry=experts[e]
                gu=read(entry['gate_up'],(1024,2048)).cuda();down=read(entry['down'],(2048,512)).cuda()
                g,u=F.linear(x[token],gu).chunk(2,-1);hidden=(F.silu(g.float())*u.float()).bfloat16()
                y=torch.mm(hidden,down.T,out_dtype=torch.float32)
                expected.index_add_(0,token,(y*w[token,slot,None]).bfloat16().float())
                del gu,down
            expected=expected.bfloat16().float().cpu()
        step={'layer':0,'ids':ids.flatten().tolist(),'weights':w.flatten().tolist(),'check_bytes':rows==1}
        baseline=None
        for ram_mib,gpu_mib in ((0,0),(12,6),(384,0),(0,384)):
            stem=f'rows{rows}-ram{ram_mib}-gpu{gpu_mib}'
            job={'index':str(ipath),'ram_bytes':ram_mib*1024**2,'vram_bytes':gpu_mib*1024**2,
                 'staging_bytes':16*1024**2,'tokens':rows,'top_k':8,'input':x.float().flatten().tolist(),
                 'steps':[step,{**step,'check_bytes':False}]}
            jp=a.output/(stem+'.json');rp=a.output/(stem+'.result.json');jp.write_text(json.dumps(job))
            result=subprocess.run([str(a.exe),str(jp),str(rp)],capture_output=True,text=True)
            (a.output/(stem+'.stderr.txt')).write_text(result.stderr)
            assert result.returncode==0,result.stderr
            actual=json.loads(rp.read_text());errors=[]
            for s in actual['results']:
                got=torch.tensor(s['output']).reshape(rows,2048);err=(got-expected).abs()
                rms=((got-expected).square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-12)).item()
                maximum=err.max().item();passed=rms<0.003 and maximum<=0.012*expected.abs().max().item()+1e-6
                errors.append({'max_abs_error':maximum,'mean_abs_error':err.mean().item(),'relative_rms_error':rms,'different_components':int((got!=expected).sum()),'passed':passed})
                if baseline is None:baseline=got
                else:assert torch.equal(got,baseline),'Cache hit/miss changed official expert output'
                if not passed:report['failures'].append({'case':stem,**errors[-1]})
            report['cases'].append({'case':stem,'unique_experts':len(set(step['ids'])),'errors':errors,'stats':actual['results'][-1]['stats']})
            print(json.dumps({'case':stem,'unique_experts':len(set(step['ids'])),'max_abs_error':max(x['max_abs_error'] for x in errors)}),flush=True)
    report['passed']=not report['failures'];(a.output/'summary.json').write_text(json.dumps(report,indent=2))
    assert report['passed'],report['failures']
    print('PASS selected official BF16 experts; full model not exercised',flush=True)


if __name__=='__main__':main()
