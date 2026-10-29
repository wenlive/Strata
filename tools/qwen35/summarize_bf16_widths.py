"""Collect Clover width controls and assert their token IDs against a causal run."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True, help='Root containing width*/ result directories')
    p.add_argument('--causal-run', type=Path, required=True)
    a = p.parse_args()
    base = json.loads((a.causal_run / 'result.json').read_text())
    base_config = json.loads((a.causal_run / 'config.json').read_text())
    result = {'physical_ssd_bytes': None, 'causal_run': str(a.causal_run),
              'note': 'Short sequential observations; first/warm means application cache, not cold SSD.', 'cases': {}}
    for root in sorted(a.runs.glob('width*')):
        if not root.is_dir():
            continue
        cfg = json.loads((root / 'config.json').read_text())
        raw = json.loads((root / 'result.json').read_text())
        summary = json.loads((root / 'benchmark-summary.json').read_text())
        assert cfg['accept_rule'] == 'exact'
        for key in ('input_tokens', 'ram_cache_mib', 'vram_cache_mib', 'staging_mib',
                    'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'max_new_tokens'):
            assert cfg[key] == base_config[key], key
        assert len(raw['runs']) == len(base['runs']) == len(summary['runs'])
        comparisons = [x['output_tokens'] == y['output_tokens'] for x, y in zip(raw['runs'], base['runs'])]
        assert all(comparisons), f'Causal/exact token mismatch: {root}'
        result['cases'][root.name] = {
            'causal_outputs_equal': comparisons, 'config': cfg, 'result': raw,
            'request': json.loads((root / 'request.json').read_text()),
            'source': json.loads((root / 'source-provenance.json').read_text()), 'summary': summary}
        print(root.name, [round(x['steady_token_per_second'], 3) for x in raw['runs']])
    assert result['cases'], 'No width runs found'
    (a.runs / 'width-summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
