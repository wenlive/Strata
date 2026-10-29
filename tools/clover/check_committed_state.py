"""Replay a native run's exact committed token sequence causally and compare state.

No rejected candidates or pending bonus are replayed. A new native process and
cache configuration can force misses independently of the original generation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--prefill-chunk',type=int,default=1)
    p.add_argument('--ram-cache-mib',type=int,default=0);p.add_argument('--vram-cache-mib',type=int,default=1024)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.run/'config.json').read_text());run=json.loads((a.run/'result.json').read_text())['runs'][a.repeat]
    assert run['committed_prefix_length']==len(run['committed_token_ids'])
    original=Path(run['state_directory'])
    cfg={'index':source['index'],'input_tokens':run['committed_token_ids'],'mode':'forward',
         'context_tokens':source['context_tokens'],'prefill_chunk':a.prefill_chunk,
         'ram_cache_mib':a.ram_cache_mib,'vram_cache_mib':a.vram_cache_mib,'staging_mib':16,
         'dump_final_state':str(a.output/'replayed-state'),'result_file':str(a.output/'result.json')}
    path=a.output/'config.json';path.write_text(json.dumps(cfg,indent=2))
    env=os.environ.copy();cuda=env.get('CUDA_PATH')
    if cuda:env['PATH']=str(Path(cuda)/'bin/x64')+os.pathsep+str(Path(cuda)/'bin')+os.pathsep+env['PATH']
    with (a.output/'native.log').open('w') as log:
        result=subprocess.run([str(a.exe),'--config',str(path)],stdout=log,stderr=subprocess.STDOUT,env=env)
    assert result.returncode==0,(a.output/'native.log').read_text()[-4000:]
    replay=a.output/'replayed-state';om=json.loads((original/'manifest.json').read_text());rm=json.loads((replay/'manifest.json').read_text())
    assert om['prefix_tokens']==rm['prefix_tokens']==len(run['committed_token_ids'])
    report={'kind':'native-generation-commit-vs-direct-causal-replay','generation_mode':run['mode'],'committed_tokens':run['committed_token_ids'],
            'prefill_chunk':a.prefill_chunk,'comparisons':[],'failures':[]}

    def values(path):
        if path.suffix=='.f32':return np.fromfile(path,dtype='<f4')
        bits=np.fromfile(path,dtype='<u2').astype(np.uint32)<<16
        return bits.view(np.float32)

    for layer in om['layers']:
        for name,desc in layer.items():
            if name=='layer':continue
            other=rm['layers'][layer['layer']][name];assert desc['bytes']==other['bytes']
            one=values(original/desc['file']);two=values(replay/other['file']);assert one.shape==two.shape
            delta=np.abs(one-two);rms=float(np.sqrt(np.mean((one.astype(np.float64)-two)**2))/max(np.sqrt(np.mean(one.astype(np.float64)**2)),1e-12)) if len(one) else 0
            maximum=float(delta.max()) if len(delta) else 0;mean=float(delta.mean()) if len(delta) else 0
            passed=rms<0.004 and maximum<=0.02*max(float(np.abs(one).max()) if len(one) else 0,1e-6)+1e-6
            row={'layer':layer['layer'],'state':name,'max_abs_error':maximum,'mean_abs_error':mean,
                 'relative_rms_error':rms,'different_components':int(np.count_nonzero(one!=two)),'passed':passed}
            with (original/desc['file']).open('rb') as file:row['original_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
            with (replay/other['file']).open('rb') as file:row['replayed_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
            row['byte_equal']=row['original_sha256']==row['replayed_sha256']
            report['comparisons'].append(row)
            if not passed:report['failures'].append(row)
    report['passed']=not report['failures'];(a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':report['passed'],'state_tensors':len(report['comparisons']),'failures':report['failures']},indent=2))
    assert report['passed']


if __name__=='__main__':main()
