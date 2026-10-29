"""Check Qwen metadata coverage and reject corrupt layouts without payload loads."""
import argparse
import copy
import json
from pathlib import Path
from index_bf16 import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--headers',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=json.loads((a.model/'config.json').read_text());c=dict(raw['text_config']);c['num_shared_experts']=1
    headers=json.loads(a.headers.read_text());wm=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    tensors,experts,excluded,totals=validate(c,headers,wm)
    assert len(experts)==10240 and totals['routed_experts']==60*1024**3
    assert sum(d['bytes'] for d in tensors.values())==69321221376
    assert len(tensors)+len(excluded)==len(wm)
    report={'scope':'metadata-only validation; no full-model execution','valid_tensors':len(tensors),'excluded_vision_mtp_tensors':len(excluded),'expert_slices':len(experts),'capacity_bytes':totals,'rejected':[]}
    name=next(n for n in wm if n.endswith('layers.0.input_layernorm.weight'));filename=wm[name]
    for kind in ('missing','dtype','shape','outside_file','wrong_weight_map','duplicate','unexpected','overlap'):
        h=copy.deepcopy(headers);w=dict(wm);t=h[filename]['header'][name]
        if kind=='missing':del h[filename]['header'][name]
        elif kind=='dtype':t['dtype']='F16'
        elif kind=='shape':t['shape'][-1]+=1
        elif kind=='outside_file':t['data_offsets']=[h[filename]['file_size'],h[filename]['file_size']+4096]
        elif kind=='wrong_weight_map':w[name]=next(f for f in h if f!=filename)
        elif kind=='duplicate':h[next(f for f in h if f!=filename)]['header'][name]=copy.deepcopy(t)
        elif kind=='unexpected':h[filename]['header']['unexpected.weight']=copy.deepcopy(t);w['unexpected.weight']=filename
        else:
            other=next(v for n,v in h[filename]['header'].items() if n!=name and n!='__metadata__' and v['shape']==t['shape'])
            t['data_offsets']=other['data_offsets'][:]
        try:validate(c,h,w)
        except ValueError as error:report['rejected'].append({'case':kind,'error':str(error)})
        else:raise AssertionError('Accepted corrupt '+kind)
    report['passed']=True;a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
