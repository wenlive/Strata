"""Summarize measured Qwen/Clover control runs; no inferred physical SSD traffic."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
    report={'scope':'complete BF16 native Windows comparison; short controlled requests','cases':{},'physical_ssd_bytes':None,'notes':['First/warm denotes application-cache state, not measured cold SSD.','Steady rate excludes first token; decode bytes below also exclude prefill.','Host timing sections are separate from GPU elapsed time; do not add GPU copies a second time.','Model numerical validation limits must be read separately.']}
    prompt=None;table=[]
    for path in sorted(a.runs.glob('*/benchmark-summary.json')):
        root=path.parent;b=json.loads(path.read_text());raw=json.loads((root/'result.json').read_text());cfg=json.loads((root/'config.json').read_text())
        assert (cfg.get('presence_penalty'),cfg.get('frequency_penalty'),cfg.get('repetition_penalty'))==(0,0,1)
        if prompt is None:prompt=cfg['input_tokens']
        assert cfg['input_tokens']==prompt,'Different prompt IDs in control comparison'
        case={'revision':raw['revision'],'config':cfg,'source':json.loads((root/'source-provenance.json').read_text()),'memory':b.get('sampled_process_and_device_memory'),'load':b['load'],'samples':[]}
        for original,s in zip(raw['runs'],b['runs']):
            phases={k:v for k,v in s['phases'].items() if k!='prefill'};tokens=len(original['output_tokens'])-1
            keys=('elapsed_ms','unique_expert_evaluations','logical_file_read_bytes','h2d_weight_bytes','d2h_weight_bytes','host_copy_bytes','host_copy_ms','read_ms','host_staging_wait_ms','ram_staging_wait_ms','demotion_wait_ms','gpu_hits','gpu_misses','ram_hits','ram_misses','read_calls')
            sums={k:sum(v[k] for v in phases.values()) for k in keys if all(k in v for v in phases.values())}
            sample={'repeat':original['repeat'],'output_ids':original['output_tokens'],'output_count':len(original['output_tokens']),'decoded_text':(root/f'completion-{original["repeat"]}.txt').read_text(),'ttft_ms':original['ttft_ms'],'generation_elapsed_ms':original['generation_elapsed_ms'],'steady_token_per_second':original['steady_token_per_second'],'decode_wall_ms':original['generation_elapsed_ms']-original['ttft_ms'],'decode_phase_totals':sums,'phases':s['phases'],'rounds':original['rounds']}
            for key in ('logical_file_read_bytes','h2d_weight_bytes','d2h_weight_bytes','host_copy_bytes'):
                sample['decode_'+key+'_per_new_token']=sums.get(key,0)/tokens if tokens>0 else None
            case['samples'].append(sample)
        case['repeat_outputs_equal']=all(x['output_ids']==case['samples'][0]['output_ids'] for x in case['samples']);assert case['repeat_outputs_equal']
        report['cases'][root.name]=case
        warm=case['samples'][-1];d=warm['decode_phase_totals']
        table.append(f"| {root.name} | {warm['steady_token_per_second']:.3f} | {warm['ttft_ms']/1000:.3f} | {d['logical_file_read_bytes']/2**30:.3f} | {d['h2d_weight_bytes']/2**30:.3f} | {d['host_copy_ms']/1000:.3f} | {d['read_ms']/1000:.3f} |")
    assert report['cases'],'No completed comparison cases'
    equal=[]
    for name,case in report['cases'].items():
        if not name.startswith('clover-causal-'):continue
        other=report['cases'][name.replace('clover-causal-','clover-selfspec-',1)]
        same=all(x['output_ids']==y['output_ids'] for x,y in zip(case['samples'],other['samples']));equal.append({'budget':name.removeprefix('clover-causal-'),'same_ids':same})
    report['clover_causal_vs_exact_output_checks']=equal;report['prompt_ids']=prompt
    (a.runs/'comparison-summary.json').write_text(json.dumps(report,indent=2))
    text='# BF16 controlled comparison\n\nWarm samples, short Count prompt, greedy, penalties 0/0/1. See JSON for both requests, source hashes and validation limits.\n\n| Case | Steady token/s | TTFT s | Decode logical reads GiB | Decode H2D GiB | Host memcpy s | File wait s |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'+'\n'.join(table)+'\n\nLogical reads are not measured physical SSD bytes.\n'
    (a.runs/'comparison-summary.md').write_text(text);print(text)
    assert all(x['same_ids'] for x in equal),'Clover exact output differs from causal under penalty-free control'


if __name__=='__main__':main()
