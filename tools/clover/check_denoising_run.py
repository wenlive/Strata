"""Check full-model denoising traces, output truncation and committed-state proof."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',type=Path)
    a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());result=json.loads((a.run/'result.json').read_text())
    index=json.loads(Path(cfg['index']).read_text());mask=index['config']['mask_token_id']
    gen=json.loads((Path(index['model_root'])/'generation_config.json').read_text());stops=set(gen['eos_token_id'])
    checks=[]
    for run in result['runs']:
        assert run['mode']=='denoise' and run['accept_rule'] is None and run['runtime_draft_width'] is None
        width=run['denoise_block_size'];threshold=run['denoise_threshold'];emitted=[];committed=list(cfg['input_tokens'])
        for number,block in enumerate(run['rounds']):
            canvas=[mask]*width;pending=set(range(width));assert 0<len(block['steps'])<=width
            for step in block['steps']:
                assert step['input_block']==canvas and len(step['predictions'])==len(step['confidence'])==width
                expected=[i for i in sorted(pending) if step['confidence'][i]>threshold]
                if not expected:expected=[max(sorted(pending),key=lambda i:step['confidence'][i])]
                assert step['transferred_positions']==expected
                for i in expected:
                    assert i in pending and step['predictions'][i]!=mask
                    canvas[i]=step['predictions'][i];pending.remove(i)
                assert step['pending_after']==len(pending)
            assert not pending and canvas==block['resolved_block']
            output=[]
            for token in canvas:
                output.append(token)
                if len(emitted)+len(output)==cfg['max_new_tokens'] or token in stops:break
            assert len(output)==block['actual_output_tokens'];emitted+=output
            continuation=number+1<len(run['rounds'])
            assert block['committed_input_tokens']==(len(output) if continuation else 0)
            if continuation:committed+=output
        assert emitted==run['output_tokens'] and committed==run['committed_token_ids']
        assert run['committed_prefix_length']==len(committed)
        assert run['uncommitted_output_suffix']==emitted[len(committed)-len(cfg['input_tokens']):]
        if len(run['rounds'])==1:assert run['steady_token_per_second'] is None
        if emitted[-1] not in stops:assert len(emitted)==cfg['max_new_tokens']
        checks.append({'repeat':run['repeat'],'blocks':len(run['rounds']),'preview_calls':sum(x['denoise_steps'] for x in run['rounds']),
                       'outputs':len(emitted),'committed_response_tokens':len(committed)-len(cfg['input_tokens']),
                       'full_canvas_verified':True,'preview_state_audit_enabled':cfg['audit_state'],'passed':True})
    report={'scope':__doc__,'native_HF_model_numerical_parity':'unresolved; this checks generation bookkeeping, not cross-engine logits','runs':checks}
    if a.replay:
        state=json.loads((a.replay/'summary.json').read_text());assert state['passed'] and all(x['byte_equal'] for x in state['comparisons'])
        report['state_tensors_sha256_equal']=len(state['comparisons'])
    (a.run/'denoising-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
