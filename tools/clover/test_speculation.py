"""Clover exact acceptance/EOS/truncation decisions against pinned SGLang functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--sglang-utils',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    code=a.sglang_utils.read_text();nodes=[x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef)]
    namespace={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(a.sglang_utils),'exec'),namespace)
    cases=[];expected=[]
    for width in (2,8,16,32):
        for count in sorted({0,width//2,width-1}):
            for stop_kind in ('none','draft','bonus'):
                for remaining in sorted({1,max(1,width//2),width,width+1}):
                    vocab=64;candidates=torch.arange(1,width+1,dtype=torch.long)[None]
                    logits=torch.zeros(1,width,vocab)
                    for row in range(width):logits[0,row,(row+2) if row<count else 47]=4
                    correct=namespace['clover_correct_drafts'](logits,candidates,accept_rule='exact')
                    assert correct.item()==count
                    bonus=logits[0,count].argmax().reshape(1)
                    output=namespace['clover_accept_tokens'](candidates,correct,bonus)[0,:count+1].tolist()
                    eos=([int(candidates[0,1])] if stop_kind=='draft' else [int(bonus[0])] if stop_kind=='bonus' else [])
                    truncated=[]
                    for token in output[:remaining]:
                        truncated.append(token)
                        if token in eos:break
                    drafts=min(count,len(truncated));emitted_bonus=len(truncated)>count
                    expected.append({'matched_drafts':count,'bonus':int(bonus[0]),'output':truncated,
                                     'commit_inputs':candidates[0,:1+drafts].tolist(),'emitted_drafts':drafts,
                                     'emitted_bonus':emitted_bonus,'pending':int(bonus[0]) if emitted_bonus else -1})
                    cases.append({'candidates':candidates.flatten().tolist(),'logits':logits.flatten().tolist(),
                                  'vocab':vocab,'remaining':remaining,'eos':eos})
    job=a.output/'job.json';job.write_text(json.dumps({'speculation':cases}))
    native=a.output/'native';r=subprocess.run([str(a.exe),str(job),str(native)],capture_output=True,text=True)
    (a.output/'stdout.txt').write_text(r.stdout);(a.output/'stderr.txt').write_text(r.stderr)
    assert r.returncode==0,r.stderr
    actual=json.loads((native/'result.json').read_text())['speculation']
    failures=[{'case':n,'got':got,'expected':want} for n,(got,want) in enumerate(zip(actual,expected)) if got!=want]
    assert len(actual)==len(expected)
    report={'kind':'synthetic-exact-decision','passed':not failures,'cases':len(actual),'failures':failures,
            'reference':str(a.sglang_utils),'reference_sha256':hashlib.sha256(code.encode()).hexdigest()}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert not failures


if __name__=='__main__':main()
