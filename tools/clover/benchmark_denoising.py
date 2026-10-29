"""Measure complete native BF16 block denoising and causal/selfspec controls.

Sequential requests only; prepare models/builds and stop downloads beforehand.
Greedy, thinking off, penalties 0/0/1; no audit/trace. No physical SSD claim.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def summarize(root):
    report={'scope':__doc__,'cases':{},'physical_ssd_bytes':None,
            'rate_note':'E2E includes prefill, excludes opening/CUDA init. Post-prefill counts all denoise tokens, but excludes the causal first token produced by prefill. Denoise steady excludes the whole first block and is null for a one-block request.'}
    rows=[];prompt=None
    for path in sorted(root.glob('*/benchmark-summary.json')):
        case=path.parent;b=json.loads(path.read_text());raw=json.loads((case/'result.json').read_text());cfg=json.loads((case/'config.json').read_text())
        if prompt is None:prompt=cfg['input_tokens']
        assert cfg['input_tokens']==prompt
        assert (cfg['presence_penalty'],cfg['frequency_penalty'],cfg['repetition_penalty'])==(0,0,1)
        samples=[]
        for r,s in zip(raw['runs'],b['runs']):
            phases={k:v for k,v in s['phases'].items() if k!='prefill'}
            sums={k:sum(v.get(k,0) for v in phases.values()) for k in ('elapsed_ms','unique_expert_evaluations','logical_file_read_bytes','h2d_weight_bytes','d2h_weight_bytes','host_copy_bytes','host_copy_ms','read_ms','demotion_wait_ms','ram_staging_wait_ms','gpu_hits','gpu_misses','ram_hits','ram_misses','read_calls')}
            denoise=r['mode']=='denoise';count=len(r['output_tokens']);post_count=count if denoise else count-1
            post_ms=r.get('decode_wall_ms',r['generation_elapsed_ms']-r['ttft_ms'])
            samples.append({'repeat':r['repeat'],'output_count':count,'output_ids':r['output_tokens'],
                'decoded_text':(case/f'completion-{r["repeat"]}.txt').read_text(encoding='utf-8'),
                'ttft_ms':r['ttft_ms'],'generation_elapsed_ms':r['generation_elapsed_ms'],
                'end_to_end_token_per_second':count*1000/r['generation_elapsed_ms'],
                'post_prefill_output_count':post_count,'post_prefill_wall_ms':post_ms,
                'post_prefill_token_per_second':post_count*1000/post_ms if post_count else None,
                'native_steady_token_per_second':r['steady_token_per_second'],'rounds':r['rounds'],'phases':s['phases'],'post_prefill_totals':sums,
                'logical_read_bytes_per_post_prefill_token':sums['logical_file_read_bytes']/post_count if post_count else None})
        assert samples and all(s['output_ids']==samples[0]['output_ids'] for s in samples),'Repeat output mismatch'
        report['cases'][case.name]={'revision':raw['revision'],'config':cfg,'request':json.loads((case/'request.json').read_text()),
            'source':json.loads((case/'source-provenance.json').read_text()),'load':b['load'],
            'sampled_memory':b.get('sampled_process_and_device_memory'),'samples':samples,
            'native_phase_records':[json.loads(line[6:]) for line in (case/'native.log').read_text(encoding='utf-8').splitlines() if line.startswith('PHASE ')]}
        s=samples[-1];rounds=s['rounds'];steps=[r['denoise_steps'] for r in rounds if 'denoise_steps' in r]
        rate=f"{s['post_prefill_token_per_second']:.3f}" if s['post_prefill_token_per_second'] is not None else '-'
        rows.append(f"| {case.name} | {s['output_count']} | {s['ttft_ms']/1000:.3f} | {s['end_to_end_token_per_second']:.3f} | {rate} | {','.join(map(str,steps)) or '-'} | {s['post_prefill_totals']['logical_file_read_bytes']/2**30:.3f} |")
    assert report['cases'],'No completed benchmark cases'
    first=next(iter(report['cases'].values()))['samples']
    report['same_output_ids_across_modes']={name:[x['output_ids']==y['output_ids'] for x,y in zip(c['samples'],first)] for name,c in report['cases'].items()}
    if 'clover-causal' in report['cases'] and 'clover-selfspec8' in report['cases']:
        report['causal_vs_selfspec_exact_same_ids']=all(x['output_ids']==y['output_ids'] for x,y in zip(report['cases']['clover-causal']['samples'],report['cases']['clover-selfspec8']['samples']))
        assert report['causal_vs_selfspec_exact_same_ids']
    text='# Native BF16 block-denoising control\n\nWarm requests, greedy, penalties0/0/1; see JSON for both requests and source hashes.\n\n| Case | Actual tokens | TTFT s | E2E token/s | Post-prefill token/s | Denoise steps/block | Post-prefill logical reads GiB |\n| --- | ---: | ---: | ---: | ---: | --- | ---: |\n'+'\n'.join(rows)+'\n\nE2E includes prefill, excludes model opening. Block-denoising output need not equal causal. No physical SSD or cold-cache claim.\n'
    (root/'denoising-summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');(root/'denoising-summary.md').write_text(text,encoding='utf-8');print(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path);p.add_argument('--index',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt',default='Explain why the sky is blue in two sentences.')
    p.add_argument('--max-new-tokens',type=int,default=64);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--widths',type=int,nargs='+',choices=(1,2,4,8,16,32),default=[8,16,32]);p.add_argument('--threshold',type=float,default=.95)
    p.add_argument('--ram-cache-mib',type=int,default=8192);p.add_argument('--vram-cache-mib',type=int,default=8192)
    p.add_argument('--staging-mib',type=int,default=16);p.add_argument('--context-tokens',type=int,default=256)
    p.add_argument('--skip-controls',action='store_true');p.add_argument('--qwen-source',type=Path);p.add_argument('--qwen-exe',type=Path);p.add_argument('--qwen-index',type=Path)
    p.add_argument('--summary-only',action='store_true',help='Summarize completed case directories; do not run inference')
    a=p.parse_args()
    if a.summary_only:summarize(a.output);return
    if not a.exe or not a.index:p.error('--exe and --index are required for inference')
    if len(set(a.widths))!=len(a.widths):p.error('Duplicate widths would overwrite evidence')
    qwen=(a.qwen_source,a.qwen_exe,a.qwen_index)
    if any(qwen) and not all(qwen):p.error('Provide all three Qwen paths')
    tools=Path(__file__).resolve().parent
    cases=[] if a.skip_controls else [('clover-causal',tools/'chat.py',a.exe,a.index,'causal',[],tools/'report_run.py'),('clover-selfspec8',tools/'chat.py',a.exe,a.index,'selfspec',['--draft-width','8','--accept-rule','exact'],tools/'report_run.py')]
    if all(qwen):cases.append(('qwen-causal',a.qwen_source/'tools/qwen35/chat_bf16.py',a.qwen_exe,a.qwen_index,'causal',[],a.qwen_source/'tools/qwen35/report_bf16_run.py'))
    cases += [(f'clover-denoise{w}',tools/'chat.py',a.exe,a.index,'denoise',['--denoise-block-size',str(w),'--denoise-threshold',str(a.threshold)],tools/'report_run.py') for w in a.widths]
    for _,chat,exe,index,_,_,report in cases:
        for path in (chat,exe,index,report):
            if not path.is_file():p.error(f'Missing input: {path}')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'benchmark-command.json').write_text(json.dumps(sys.argv,indent=2),encoding='utf-8')
    for name,chat,exe,index,mode,extra,report in cases:
        dest=a.output/name
        command=[sys.executable,'-u',str(chat),'--exe',str(exe.resolve()),'--index',str(index.resolve()),'--output',str(dest.resolve()),'--prompt',a.prompt,'--mode',mode,
                 '--max-new-tokens',str(a.max_new_tokens),'--context-tokens',str(a.context_tokens),'--ram-cache-mib',str(a.ram_cache_mib),'--vram-cache-mib',str(a.vram_cache_mib),
                 '--staging-mib',str(a.staging_mib),'--prefill-chunk','32','--repeats',str(a.repeats),'--presence-penalty','0','--frequency-penalty','0','--repetition-penalty','1']+extra
        with (a.output/f'{name}.log').open('w',encoding='utf-8') as log:
            log.write(json.dumps(command)+'\n');log.flush();r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise SystemExit(f'Exit {r.returncode}: {a.output/name}.log')
        with (a.output/f'{name}-summary.log').open('w',encoding='utf-8') as log:r=subprocess.run([sys.executable,str(report),'--run',str(dest)],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise SystemExit(f'Report failed: {dest}')
        print(f'Completed {name}',flush=True)
    summarize(a.output)


if __name__=='__main__':main()
