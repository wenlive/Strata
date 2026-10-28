"""Tokenize with the official template, then run Strata's native BF16 Clover CLI.

This Python entry only handles tokenizer, config and decoded text. All model
weights, expert tiering, attention/GDN and generation execute in strata-clover.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--output',type=Path,required=True,help='New evidence directory')
    p.add_argument('--mode',choices=('selfspec','causal'),default='selfspec')
    p.add_argument('--accept-rule',choices=('exact',),default='exact')
    p.add_argument('--draft-width',type=int,choices=(2,4,8,16,32),default=8)
    p.add_argument('--max-new-tokens',type=int,default=16)
    p.add_argument('--context-tokens',type=int,default=256)
    p.add_argument('--prefill-chunk',type=int,default=32)
    p.add_argument('--ram-cache-mib',type=int,default=4096)
    p.add_argument('--vram-cache-mib',type=int,default=6144)
    p.add_argument('--staging-mib',type=int,default=16)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--thinking',action='store_true',help='Enable the official thinking template (default: disabled)')
    p.add_argument('--audit-state',action='store_true')
    p.add_argument('--dump-state',action='store_true')
    p.add_argument('--trace',action='store_true',help='Write per-layer correctness traces; includes extra synchronization and I/O')
    for key in ('presence','frequency','repetition'):
        p.add_argument(f'--{key}-penalty',type=float)
    a=p.parse_args()
    index=json.loads(a.index.read_text(encoding='utf-8'))
    if not index.get('complete') or index.get('scope')!='full-model':
        p.error('A complete, verified full-model index is required')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(index['model_root'],trust_remote_code=False)
    messages=[{'role':'user','content':a.prompt}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=a.thinking)
    tokens=tokenizer(rendered,add_special_tokens=False)['input_ids']
    a.output.mkdir(parents=True,exist_ok=False)
    config={'index':str(a.index.resolve()),'input_tokens':tokens,'mode':a.mode,
            'accept_rule':a.accept_rule,'draft_width':a.draft_width,
            'max_new_tokens':a.max_new_tokens,'context_tokens':a.context_tokens,'prefill_chunk':a.prefill_chunk,
            'ram_cache_mib':a.ram_cache_mib,'vram_cache_mib':a.vram_cache_mib,
            'staging_mib':a.staging_mib,'repeats':a.repeats,'audit_state':a.audit_state,
            'result_file':str((a.output/'result.json').resolve())}
    if a.dump_state:config['dump_final_state']=str((a.output/'state').resolve())
    if a.trace:config['trace_directory']=str((a.output/'trace').resolve())
    for key in ('presence','frequency','repetition'):
        value=getattr(a,key+'_penalty')
        if value is not None:config[key+'_penalty']=value
    (a.output/'request.json').write_text(json.dumps({'messages':messages,'thinking':a.thinking,'tokenizer_root':index['model_root'],'tokenizer_class':type(tokenizer).__name__},indent=2),encoding='utf-8')
    path=a.output/'config.json';path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    source_root=Path(__file__).resolve().parents[2]
    commit=subprocess.run(['git','-C',str(source_root),'rev-parse','HEAD'],capture_output=True,text=True)
    dirty=subprocess.run(['git','-C',str(source_root),'status','--short'],capture_output=True,text=True)
    digest=hashlib.sha256()
    with a.exe.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    (a.output/'source-provenance.json').write_text(json.dumps({'source_commit':commit.stdout.strip(),
        'source_dirty':dirty.stdout.splitlines(),'native_executable':str(a.exe.resolve()),'native_executable_sha256':digest.hexdigest()},indent=2),encoding='utf-8')
    env=os.environ.copy();cuda=env.get('CUDA_PATH')
    if cuda:env['PATH']=str(Path(cuda)/'bin/x64')+os.pathsep+str(Path(cuda)/'bin')+os.pathsep+env['PATH']
    start=time.perf_counter()
    with (a.output/'native.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([str(a.exe), '--config',str(path)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',env=env)
        from telemetry import MemorySampler
        sampler=MemorySampler(proc.pid);sampler.start()
        for line in proc.stdout:
            log.write(line);log.flush()
            if line.startswith(('LOAD ','ROUND ')):print(line.rstrip(),flush=True)
            elif line.startswith('RESULT '):
                result=json.loads(line[7:]);text=tokenizer.decode(result['output_tokens'],skip_special_tokens=True)
                print(text,flush=True)
                (a.output/f'completion-{result["repeat"]}.txt').write_text(text,encoding='utf-8')
        code=proc.wait()
        memory=sampler.finish()
    (a.output/'telemetry.json').write_text(json.dumps(memory,indent=2),encoding='utf-8')
    (a.output/'process.json').write_text(json.dumps({'exit_code':code,'wall_seconds':time.perf_counter()-start,
        'command':[str(a.exe),'--config',str(path)]},indent=2),encoding='utf-8')
    if code:raise SystemExit(code)


if __name__=='__main__':
    main()
