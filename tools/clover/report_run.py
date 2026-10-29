"""Summarize measured native phase times, logical I/O, caches and observed memory.

Phase-boundary CUDA samples are explicitly observations, not a proof of the
absolute GPU peak. No logical file byte counter is labeled physical SSD I/O.
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.run/'result.json').read_text());phases=[];loads=[]
    for line in (a.run/'native.log').read_text(encoding='utf-8').splitlines():
        if line.startswith('PHASE '):phases.append(json.loads(line[6:]))
        elif line.startswith('LOAD '):loads.append(json.loads(line[5:]))
    per_run=[];at=0;first=None
    for run in result['runs']:
        # Each request starts with one or more prefill phases; subsequent
        # prefill begins the next repeated request.
        selected=[];seen_nonprefill=False
        while at<len(phases):
            phase=phases[at]
            if selected and seen_nonprefill and phase['phase']=='prefill':break
            selected.append(phase);at+=1
            if phase['phase']!='prefill':seen_nonprefill=True
            if not run['rounds'] and phase['phase']=='prefill' and phase['prefix_after']==len(run['input_tokens']):break
        by_phase=collections.defaultdict(lambda:{'calls':0,'tokens':0,'elapsed_ms':0,'logical_file_read_bytes':0,'h2d_weight_bytes':0,'d2h_weight_bytes':0,'h2d_control_bytes':0,'read_calls':0,
                                                   'gpu_hits':0,'gpu_misses':0,'ram_hits':0,'ram_misses':0,'unique_expert_evaluations':0})
        memories=[]
        for phase in selected:
            out=by_phase[phase['phase']];out['calls']+=1;out['tokens']+=phase['input_tokens'];out['elapsed_ms']+=phase['elapsed_ms']
            for key in ('logical_file_read_bytes','read_calls','gpu_hits','gpu_misses','ram_hits','ram_misses'):out[key]+=phase['cache_io'][key]
            out['h2d_weight_bytes']+=phase['cache_io']['h2d_bytes'];out['h2d_control_bytes']+=phase['control_h2d_bytes']
            out['d2h_weight_bytes']+=phase['cache_io'].get('d2h_bytes',0)
            for key in ('host_copy_bytes','host_copy_ms','demotion_wait_ms','ram_staging_wait_ms','read_ms','host_staging_wait_ms'):
                if key in phase['cache_io']:out[key]=out.get(key,0)+phase['cache_io'][key]
            out['unique_expert_evaluations']+=sum(phase['experts_unique_per_layer']);memories.append(phase['memory'])
        if first is None:first=run['output_tokens']
        summary={k:run[k] for k in ('repeat','run_kind','mode','ttft_ms','generation_elapsed_ms','steady_token_per_second','logical_expert_read_bytes_per_output_token')}
        for key in ('first_chunk_tokens','output_chunks','prefill_wall_ms','decode_wall_ms','decode_output_tokens','end_to_end_token_per_second','decode_only_token_per_second','steady_rate_definition','denoise_block_size','denoise_threshold'):
            if key in run:summary[key]=run[key]
        summary.update({'outputs':len(run['output_tokens']),'outputs_match_first':run['output_tokens']==first,'phases':dict(by_phase),'round_acceptance':run['rounds'],
            'peak_process_working_set_bytes':max((m['peak_working_set_bytes'] for m in memories),default=0),
            'peak_process_commit_bytes':max((m['peak_pagefile_bytes'] for m in memories),default=0),
            'peak_observed_device_used_bytes_at_phase_boundaries':max((m['cuda_total_bytes']-m['cuda_free_bytes'] for m in memories),default=0)})
        per_run.append(summary)
    report={'revision':result['revision'],'load':loads,'runs':per_run,
            'physical_ssd_bytes':None,'io_note':'Logical payload/read calls and aligned completions only; physical SSD traffic and cold-I/O state are unmeasured.',
            'vram_note':'CUDA free-memory observations at phase boundaries include device/driver/desktop use; not a continuous per-process peak measurement.'}
    telemetry=a.run/'telemetry.json'
    if telemetry.exists():report['sampled_process_and_device_memory']=json.loads(telemetry.read_text())
    (a.run/'benchmark-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
