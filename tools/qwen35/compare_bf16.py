"""Run native Qwen/Clover BF16 controls sequentially; preserve every command/log.

Build both independent branches first. No model downloads or HF model inference
are performed here. Both checkpoints use their official non-thinking template.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qwen-exe', 'qwen-index', 'clover-source', 'clover-exe', 'clover-index', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prompt', default='Count from 1 to 10.')
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--context-tokens', type=int, default=256)
    p.add_argument('--prefill-chunk', type=int, default=32)
    p.add_argument('--staging-mib', type=int, default=16)
    p.add_argument('--draft-width', type=int, choices=(2, 4, 8, 16, 32), default=8)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--budget', type=int, nargs=2, action='append', metavar=('RAM_MIB', 'GPU_MIB'),
                   help='Repeatable expert-cache budgets; default: 0/6144, 8192/6144, 8192/8192')
    a = p.parse_args()
    budgets = a.budget or [(0, 6144), (8192, 6144), (8192, 8192)]
    if len({tuple(x) for x in budgets}) != len(budgets):
        p.error('Duplicate budgets would overwrite evidence')
    if any(r < 0 or g < 0 for r, g in budgets) or a.repeats < 1:
        p.error('Cache budgets must be nonnegative and repeats positive')
    qwen_tools = Path(__file__).resolve().parent
    clover_tools = a.clover_source.resolve() / 'tools' / 'clover'
    variants = [
        ('qwen-causal', qwen_tools / 'chat_bf16.py', a.qwen_exe, a.qwen_index, 'causal', qwen_tools / 'report_bf16_run.py'),
        ('clover-causal', clover_tools / 'chat.py', a.clover_exe, a.clover_index, 'causal', clover_tools / 'report_run.py'),
        ('clover-selfspec', clover_tools / 'chat.py', a.clover_exe, a.clover_index, 'selfspec', clover_tools / 'report_run.py'),
    ]
    for _, chat, exe, index, _, report in variants:
        for path in (chat, exe, index, report):
            if not path.is_file():
                p.error(f'Missing input: {path}')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'comparison-command.json').write_text(json.dumps(sys.argv, indent=2), encoding='utf-8')

    def run(command, log):
        with log.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(command) + '\n')
            stream.flush()
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Exit {result.returncode}; retained evidence: {log}')

    for ram, gpu in budgets:
        for name, chat, exe, index, mode, report in variants:
            case = f'{name}-ram{ram}-gpu{gpu}'
            output = a.output / case
            command = [sys.executable, '-u', str(chat), '--exe', str(exe.resolve()),
                       '--index', str(index.resolve()), '--output', str(output.resolve()),
                       '--prompt', a.prompt, '--mode', mode,
                       '--presence-penalty', '0', '--frequency-penalty', '0', '--repetition-penalty', '1']
            for key in ('max_new_tokens', 'context_tokens', 'prefill_chunk', 'staging_mib', 'repeats'):
                command += ['--' + key.replace('_', '-'), str(getattr(a, key))]
            command += ['--ram-cache-mib', str(ram), '--vram-cache-mib', str(gpu)]
            if mode == 'selfspec':
                command += ['--draft-width', str(a.draft_width), '--accept-rule', 'exact']
            run(command, a.output / f'{case}.log')
            run([sys.executable, str(report), '--run', str(output)], a.output / f'{case}-summary.log')
            print(f'Completed {case}', flush=True)
    run([sys.executable, str(qwen_tools / 'summarize_bf16_comparison.py'), '--runs', str(a.output)],
        a.output / 'summary.log')
    print(f'Results: {a.output / "comparison-summary.md"}', flush=True)


if __name__ == '__main__':
    main()
