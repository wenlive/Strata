"""Check production LowConfidence selection/penalties against pinned HF Torch.

Synthetic controller tests, including actual 248320-wide vocabulary logits.
This is separate from complete-checkpoint generation and state tests.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--hf-sampling',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('official_clover_sampling',a.hf_sampling)
    hf=importlib.util.module_from_spec(spec);spec.loader.exec_module(hf)
    torch.manual_seed(2931);cases=[];references=[]
    for i in range(50):
        vocab=248320 if i>=46 else 16;rows=32 if i==47 else (8 if i>=46 else (1,2,4,8)[i%4])
        logits=(torch.randn(rows,vocab)*3).bfloat16().float()
        block=torch.zeros(rows,dtype=torch.int64);history=[]
        if i%3==1 and rows>1:block[0]=4
        if i%3==2:history=[4,4,5]
        gen=SimpleNamespace(**(hf.PENALTY_DEFAULTS|{'dllm_think_end_token_id':3,'do_sample':False}))
        if i%4 in (1,2):
            gen.presence_penalty=.9;gen.frequency_penalty=.2;gen.repetition_penalty=1.05
            gen.dllm_length_penalty=2.;gen.dllm_length_penalty_start=0;gen.dllm_length_penalty_window=4
            gen.dllm_think_end_bias=1.5;gen.dllm_think_end_start=0;gen.dllm_think_end_window=4
        if i%8==5:history.append(3)
        if i%8==6 and rows>1:block[0]=3
        threshold=(0.,.5,.95,1.)[i%4]
        # Exact confidence tie: 0.5 must not pass a strict 0.5 threshold.
        if i in (40,41):
            logits.fill_(-1000);logits[:,4]=0;logits[:,5]=0;threshold=.5
            gen=SimpleNamespace(**(hf.PENALTY_DEFAULTS|{'dllm_think_end_token_id':3,'do_sample':False}));history=[];block.zero_()
        if i>=48:
            target=.9499 if i==48 else .9501;logits.zero_();logits[:,4]=np.log((vocab-2)*target/(1-target));threshold=.95
            gen=SimpleNamespace(**(hf.PENALTY_DEFAULTS|{'dllm_think_end_token_id':3,'do_sample':False}));history=[];block.zero_()
        raw=a.output/f'logits{i}.f32';penalized=a.output/f'penalized{i}.f32';logits.numpy().tofile(raw)
        cfg={'eos_token_id':[1,2],'pad_token_id':2,**vars(gen)}
        resolved=[5,6,1,7,2,8];remaining=(1,2,6)[i%3]
        cases.append({'vocab':vocab,'mask':0,'block':block.tolist(),'history':history,'generation_config':cfg,
                      'threshold':threshold,'logits_file':str(raw),'penalized_file':str(penalized),'resolved_block':resolved,'remaining':remaining})
        logits[:,0]=-float('inf');state=hf.CloverPenaltyState(gen,vocab,'cpu',[1,2])
        if history:state.commit(torch.tensor(history))
        state.apply(logits,block[None],0)
        # The actual HF baseline executes softmax on CUDA. Torch CPU softmax
        # has a separately measured ~5e-6 error at this large vocabulary.
        prediction,confidence=hf.predict_with_confidence(logits.cuda(),gen)
        prediction=prediction.cpu();confidence=confidence.cpu()
        masked_conf=confidence.masked_fill(block!=0,-float('inf'))
        transfer=(masked_conf>threshold).nonzero().flatten()
        if not len(transfer):transfer=masked_conf.argmax().reshape(1)
        updated=block.clone();updated[transfer]=prediction[transfer]
        prefix=[]
        for token in resolved:
            prefix.append(token)
            if len(prefix)==remaining or token in (1,2):break
        truth=logits.double().softmax(-1).max(-1).values.float()
        references.append((logits, prediction.tolist(),confidence,truth,transfer.tolist(),updated.tolist(),prefix))
    job=a.output/'job.json';job.write_text(json.dumps({'cases':cases},indent=2))
    output=a.output/'native.json';proc=subprocess.run([str(a.exe),str(job),str(output)],capture_output=True,text=True)
    (a.output/'native.log').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stderr
    actual=json.loads(output.read_text());report={'scope':__doc__,
        'confidence_oracle':'FP64 mathematical softmax accuracy <=3e-7; native runtime uses compensated FP32 only. HF CUDA predicted IDs and transfer decisions must match exactly.',
        'cuda_note':'CUDA FP32 reduction itself differs from FP64 by ~6e-6 on large-vocabulary high-confidence flat tails; preserve measured deltas rather than hiding them in a widened bound.',
        'cases':[],'failures':[]}
    for i,(row,ref,case) in enumerate(zip(actual,references,cases)):
        logits,prediction,confidence,truth,transfer,updated,prefix=ref
        observed=torch.from_numpy(np.fromfile(case['penalized_file'],dtype='<f4').reshape(logits.shape))
        finite=logits.isfinite();pe=float((logits[finite]-observed[finite]).abs().max())
        ce=float((confidence-torch.tensor(row['confidence'])).abs().max())
        accuracy=float((truth-torch.tensor(row['confidence'])).abs().max())
        passed=(pe<=2e-6 and accuracy<=3e-7 and torch.equal(logits.isneginf(),observed.isneginf()) and
                row['prediction']==prediction and row['transfer']==transfer and row['updated_block']==updated and row['output_prefix']==prefix)
        item={'case':i,'rows':len(case['block']),'vocab':case['vocab'],'threshold':case['threshold'],
              'penalty_max_abs_error':pe,'confidence_cuda_max_abs_error':ce,'confidence_fp64_max_abs_error':accuracy,
              'cuda_fp64_max_abs_error':float((truth-confidence).abs().max()),'passed':passed}
        report['cases'].append(item)
        if not passed:report['failures'].append(item)
    report['invalid_inputs']=[]
    for name,change in [('threshold-negative',{'threshold':-.1}),('threshold-high',{'threshold':1.1}),('all-resolved',{'block':[4]}),('truncated-logits',{'logits_file':str(a.output/'empty.f32')})]:
        (a.output/'empty.f32').write_bytes(b'')
        bad=cases[0]|change;path=a.output/(name+'.json');path.write_text(json.dumps({'cases':[bad]}))
        proc=subprocess.run([str(a.exe),str(path),str(a.output/(name+'-out.json'))],capture_output=True,text=True)
        report['invalid_inputs'].append({'name':name,'exit_code':proc.returncode,'error':proc.stderr.strip()});assert proc.returncode!=0
    report['passed']=not report['failures'];(a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'invalid_inputs':len(report['invalid_inputs']),
                      'confidence_cuda_max_abs_error':max(x['confidence_cuda_max_abs_error'] for x in report['cases']),
                      'confidence_fp64_max_abs_error':max(x['confidence_fp64_max_abs_error'] for x in report['cases']),'failures':report['failures']},indent=2))
    assert report['passed']


if __name__=='__main__':main()
