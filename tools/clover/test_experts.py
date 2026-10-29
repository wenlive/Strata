"""GPU numerical and tier-eviction tests for Strata Clover (synthetic, not model support).

Run using a CUDA PyTorch environment. Uses one complete synthetic expert at a
time on GPU; a sparse >4 GiB safetensors fixture checks 64-bit source offsets.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def fixture(root):
    torch.manual_seed(407)
    h,ff,ne,nl=2048,512,4,2
    weights={}
    for l in range(nl):
        pre=f'model.layers.{l}.mlp.experts.'
        weights[pre+'gate_up_proj']=(torch.randn(ne,2*ff,h)*0.03).bfloat16()
        weights[pre+'down_proj']=(torch.randn(ne,h,ff)*0.03).bfloat16()
    small=root/'ordinary.safetensors'
    save_file(weights,str(small))
    with small.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        header=json.loads(f.read(length));original_start=length+8
    padding=1<<32
    new_header=copy.deepcopy(header)
    for t in new_header.values():
        t['data_offsets']=[x+padding for x in t['data_offsets']]
    new_header['_padding']={'dtype':'U8','shape':[padding],'data_offsets':[0,padding]}
    encoded=json.dumps(new_header,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    big=root/'offset64.safetensors'
    big.touch()
    subprocess.run(['fsutil','sparse','setflag',str(big)],check=True,capture_output=True)
    data_start=len(encoded)+8
    with small.open('rb') as src,big.open('wb') as dst:
        dst.write(struct.pack('<Q',len(encoded)));dst.write(encoded)
        dst.seek(data_start+padding);src.seek(original_start)
        shutil.copyfileobj(src,dst,1024**2)
    tensors={}
    for name,t in header.items():
        start,end=t['data_offsets']
        tensors[name]={'file':big.name,'offset':data_start+padding+start,'bytes':end-start,
                       'shape':t['shape'],'dtype':t['dtype'],'layout':'row-major'}
    experts=[]
    for l in range(nl):
        gu=tensors[f'model.layers.{l}.mlp.experts.gate_up_proj']
        down=tensors[f'model.layers.{l}.mlp.experts.down_proj']
        for e in range(ne):
            experts.append({'layer':l,'expert':e,
                'gate_up':{'file':big.name,'offset':gu['offset']+e*2*ff*h*2,'bytes':2*ff*h*2},
                'down':{'file':big.name,'offset':down['offset']+e*h*ff*2,'bytes':h*ff*2}})
    index={'format':'strata-clover-bf16-index','version':1,'complete':True,
           'scope':'synthetic-expert-test','revision':'synthetic-seed-407',
           'model_root':str(root.resolve()),'config':{'hidden_size':h,'moe_intermediate_size':ff,
           'num_hidden_layers':nl,'num_experts':ne},'tensors':tensors,'experts':experts,
           'files':{big.name:{'file_size':big.stat().st_size}},
           'test_note':'Sparse U8 padding tests offsets >4 GiB; no padding is read into memory'}
    (root/'index.json').write_text(json.dumps(index,indent=2))
    # Check source BF16 slices byte-for-byte after relocation, with bounded I/O.
    with small.open('rb') as a,big.open('rb') as b:
        for name,t in tensors.items():
            original=header[name]['data_offsets'][0]+original_start
            a.seek(original);b.seek(t['offset']);left=t['bytes']
            while left:
                n=min(left,1024**2);assert a.read(n)==b.read(n);left-=n
    return weights,index


@torch.inference_mode()
def oracle(weights,x,step,top_k):
    h=x.shape[1]
    ids=torch.tensor(step['ids'],device='cuda').reshape(-1,top_k)
    routes=torch.tensor(step['weights'],device='cuda',dtype=torch.float32).reshape(-1,top_k)
    y=torch.zeros_like(x,dtype=torch.float32)
    pre=f'model.layers.{step["layer"]}.mlp.experts.'
    for expert in ids.unique(sorted=True).tolist():
        token,slot=torch.where(ids==expert)
        gu=weights[pre+'gate_up_proj'][expert].cuda()
        down=weights[pre+'down_proj'][expert].cuda()
        gate,up=F.linear(x[token],gu).chunk(2,-1)
        middle=(F.silu(gate.float())*up.float()).bfloat16()
        values=torch.mm(middle,down.T,out_dtype=torch.float32)
        routed=(values*routes[token,slot,None]).bfloat16()
        y.index_add_(0,token,routed.float())
        del gu,down
    return y.bfloat16().float().cpu()


def run(exe,path,result):
    r=subprocess.run([str(exe),str(path),str(result)],capture_output=True,text=True)
    path.with_suffix('.stdout.txt').write_text(r.stdout,encoding='utf-8')
    path.with_suffix('.stderr.txt').write_text(r.stderr,encoding='utf-8')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    weights,index=fixture(a.output)
    report={'kind':'synthetic-experts-only','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'fixture':{'hidden':2048,'ff':512,'experts':4,'layers':2,
            'small_bytes':(a.output/'ordinary.safetensors').stat().st_size,
            'large_file_bytes':(a.output/'offset64.safetensors').stat().st_size,
            'minimum_tensor_offset':min(t['offset'] for t in index['tensors'].values())},'cases':[],'failures':[]}
    baseline={}
    for rows in (1,8,16,32):
        torch.manual_seed(100+rows)
        x=torch.randn(rows,2048).bfloat16()
        ids=[(t+k)%4 for t in range(rows) for k in range(2)]
        scales=[s for t in range(rows) for s in (0.625,0.375)]
        # A token must never contain a duplicated expert; the hot steps use k=2.
        hot=[e for t in range(rows) for e in (0,1)]
        steps=[{'layer':0,'ids':hot,'weights':scales},
               {'layer':0,'ids':hot,'weights':scales},
               {'layer':1,'ids':ids,'weights':scales},
               {'layer':0,'ids':ids,'weights':scales,'check_bytes':rows==1}]
        refs=[oracle(weights,x.cuda(),step,2) for step in steps]
        for ram_slots,gpu_slots in ((0,0),(1,1),(8,8),(8,0),(0,8),(2,1)):
            job={'index':str(a.output/'index.json'),'ram_bytes':ram_slots*6*1024**2,
                 'vram_bytes':gpu_slots*6*1024**2,'staging_bytes':16*1024**2,
                 'tokens':rows,'top_k':2,'input':x.float().flatten().tolist(),'steps':steps}
            stem=f'rows{rows}-ram{ram_slots}-gpu{gpu_slots}'
            job_path=a.output/(stem+'.json');result=a.output/(stem+'.result.json')
            job_path.write_text(json.dumps(job))
            r=run(a.exe,job_path,result)
            assert r.returncode==0,(stem,r.stderr[-4000:])
            parsed=json.loads(result.read_text());errors=[]
            for step in parsed['results']:
                stats=step['stats']
                assert stats['ram_gpu_duplicate_experts']==0,'Exclusive caches contain duplicate experts'
                assert stats['host_staging_bytes']+stats['gpu_staging_bytes']<=job['staging_bytes']
            for n,(actual,ref) in enumerate(zip(parsed['results'],refs)):
                got=torch.tensor(actual['output']).reshape(rows,2048)
                diff=(got-ref).abs();maxerr=diff.max().item();meanerr=diff.mean().item()
                relative_rms=((got-ref).square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)).item()
                # BF16 rounding may move a few values by one ulp. This bound
                # checks both aggregate error and maximum error; actual values
                # and differing components are always retained for diagnosis.
                passed=relative_rms<0.003 and maxerr<=0.012*ref.abs().max().item()+1e-6
                errors.append({'step':n,'max_abs_error':maxerr,'mean_abs_error':meanerr,
                               'relative_rms_error':relative_rms,'different_components':int((got!=ref).sum()),'passed':passed})
                if not passed:report['failures'].append({'case':stem,**errors[-1]})
                key=(rows,n)
                if ram_slots==gpu_slots==0:baseline[key]=got
                else:assert torch.equal(got,baseline[key]),f'Cache-dependent output {stem} step {n}'
            case={'case':stem,'errors':errors,'final_stats':parsed['results'][-1]['stats'],
                  'peak_memory':parsed['results'][-1]['memory']}
            report['cases'].append(case)
            print(json.dumps({'case':stem,'max_abs':max(x['max_abs_error'] for x in errors),
                              'max_relative_rms':max(x['relative_rms_error'] for x in errors)}),flush=True)
    # Negative tests must reject corrupt manifests before allocating caches.
    template=json.loads((a.output/'rows1-ram0-gpu0.json').read_text())
    for corruption in ('incomplete','dtype','shape','truncated','expert_offset','staging_budget'):
        damaged=copy.deepcopy(index);job=copy.deepcopy(template)
        tensor=next(iter(damaged['tensors'].values()))
        if corruption=='incomplete':damaged['complete']=False
        elif corruption=='dtype':tensor['dtype']='F16'
        elif corruption=='shape':tensor['shape'][-1]+=1
        elif corruption=='truncated':damaged['files']['offset64.safetensors']['file_size']+=1
        elif corruption=='expert_offset':damaged['experts'][0]['gate_up']['offset']+=2
        else:job['staging_bytes']=1
        badindex=a.output/(f'bad-{corruption}-index.json');badindex.write_text(json.dumps(damaged))
        job['index']=str(badindex);jobpath=a.output/(f'bad-{corruption}.json');jobpath.write_text(json.dumps(job))
        r=run(a.exe,jobpath,a.output/(f'bad-{corruption}.result.json'))
        assert r.returncode!=0,f'Accepted corrupt {corruption}'
        report.setdefault('rejected_invalid_cases',[]).append({'case':corruption,'error':r.stderr.strip()})
    report['passed']=not report['failures']
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    assert report['passed'],report['failures']
    print('PASS all synthetic expert tests; full Clover model not exercised',flush=True)


if __name__=='__main__':
    main()
