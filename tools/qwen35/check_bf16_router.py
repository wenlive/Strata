"""Check native routing against Torch using identical saved BF16 logits.

This isolates router arithmetic from upstream projection/attention differences.
Slot order can differ at ties; expert sets and per-ID weights must be exact.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--native',type=Path,help='Selected-layer native output directory')
    source.add_argument('--trace',type=Path,help='Complete-model phase trace directory')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    with torch.inference_mode():
        paths=a.trace.glob('layer*-routes.json') if a.trace else a.native.glob('step*/trace/layer0-routes.json')
        for path in sorted(paths):
            native=json.loads(path.read_text());ids=torch.tensor(native['ids']).reshape(-1,8)
            logits_path=path.with_name(path.name.replace('-routes.json','-router.bf16'))
            logits=torch.from_numpy(np.fromfile(logits_path,dtype=np.uint16)).view(torch.bfloat16).reshape(-1,256).cuda()
            probs=logits.float().softmax(-1);weights,expected=probs.topk(8,-1);weights=(weights/weights.sum(-1,keepdim=True)).bfloat16().float().cpu();expected=expected.cpu()
            actual=torch.tensor(native['weights']).reshape(-1,8);gi,go=ids.sort(-1);ri,ro=expected.sort(-1)
            left=actual.gather(-1,go);right=weights.gather(-1,ro)
            row={'source':str(path),'step':path.parent.parent.name,'different_slots':int((ids!=expected).sum()),'different_expert_set_rows':int((gi!=ri).any(-1).sum()),'max_per_expert_weight_error':float((left-right).abs().max()),'weights_exact':bool(torch.equal(left,right))}
            row['passed']=row['different_expert_set_rows']==0 and row['weights_exact'];rows.append(row)
    report={'scope':'router-only, identical native BF16 logits','passed':bool(rows) and all(x['passed'] for x in rows),'cases':rows}
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
