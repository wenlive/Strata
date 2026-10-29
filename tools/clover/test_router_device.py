"""Compare production CUDA routing weights with HF CUDA on identical logits."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from test_official_layers import oracle_namespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hf-source',type=Path,help='Also compare real preview coreset choices on identical saved logits')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.manual_seed(2932)
    cases=[];expected=[];routing=[]
    oracle=oracle_namespace(a.hf_source) if a.hf_source else None
    for i in range(48):
        if i<40:
            bits=np.fromfile(a.trace/f'layer{i}-router.bf16',dtype='<u2').copy()
            logits=torch.from_numpy(bits).view(torch.bfloat16).reshape(-1,256)
        else:
            rows=(1,8,16,32)[i%4];logits=(torch.randn(rows,256)*3).bfloat16()
            if i in (40,41):logits.zero_()
        allowed=torch.ones(256,dtype=torch.int32)
        native_route=None;coreset_equal=None
        if i<40:
            native_route=json.loads((a.trace/f'layer{i}-routes.json').read_text())
            if native_route['coreset']:
                allowed.zero_();allowed[native_route['coreset']]=1
                if oracle:
                    masked=oracle['coreset_torch'](logits[None].cuda(),torch.ones(1,len(logits),device='cuda',dtype=torch.bool),8,.9)
                    ref_allowed=masked.isfinite()[0,0].cpu().int()
                    coreset_equal=torch.equal(allowed,ref_allowed)
        if i>=44:allowed.zero_();allowed[torch.randperm(256)[:(3 if i==44 else 80)]]=1
        x=logits.cuda().float().masked_fill(~allowed.cuda().bool()[None],-float('inf'))
        ids=x.argsort(dim=-1,descending=True,stable=True)[:,:8].int()
        weights=x.softmax(-1).gather(-1,ids.long());weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        expected.append(weights.cpu())
        if native_route:
            got_ids=torch.tensor(native_route['ids']).reshape_as(ids).cuda()
            routing.append({'layer':i,'same_input_ids_equal':bool(torch.equal(ids,got_ids)),
                            'coreset_equal':coreset_equal,
                            'native_weight_max_abs_error':float((weights.cpu()-torch.tensor(native_route['weights']).reshape_as(weights.cpu())).abs().max())})
        inputs={}
        for key,tensor in [('logits',logits.view(torch.uint16)),('ids',ids.cpu()),('allowed',allowed)]:
            path=a.output/f'{i}-{key}.bin';tensor.numpy().tofile(path);inputs[key]={'file':str(path.resolve()),'bytes':path.stat().st_size}
        cases.append({'rows':len(logits),**inputs})
    job=a.output/'job.json';job.write_text(json.dumps({'router_device':cases},indent=2))
    output=a.output/'native';r=subprocess.run([str(a.exe),str(job),str(output)],capture_output=True,text=True)
    (a.output/'native.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    results=[]
    for i,ref in enumerate(expected):
        actual=torch.from_numpy(np.fromfile(output/f'router-weights{i}.f32',dtype='<f4').copy()).reshape_as(ref)
        d=(actual-ref).abs();results.append({'case':i,'rows':len(ref),'max_abs_error':float(d.max()),'mean_abs_error':float(d.mean()),'different_fp32_weights':int((ref!=actual).sum()),'passed':bool(d.max()<=2e-7)})
    report={'scope':__doc__,'cases':results,'real_trace_routing':routing,
            'passed':all(x['passed'] for x in results) and all(x['same_input_ids_equal'] and x['coreset_equal'] is not False and x['native_weight_max_abs_error']<=2e-7 for x in routing)}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':report['passed'],'cases':len(results),'max_abs_error':max(x['max_abs_error'] for x in results),'different_real_weights':sum(x['different_fp32_weights'] for x in results[:40])},indent=2));assert report['passed']


if __name__=='__main__':main()
