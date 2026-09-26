"""Compare greedy and speculative decoding of ONE trained target checkpoint.

All decoder/precision/timing options are forwarded to the existing benchmark.
This separate entry point cannot select an original-base greedy or another
target adapter. Frozen-target experiments retain their existing entry points.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def command(argv):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args, rest = parser.parse_known_args(argv)
    if any(x.startswith('--') and any(flag.startswith(x.split('=')[0])
           for flag in ('--before', '--after')) for x in rest):
        parser.error('Use one --checkpoint for both trained-target baselines')
    checkpoint = args.checkpoint.resolve()
    for name in ('adapter_model.safetensors', 'recurft_recurrent.safetensors', 'recurft_config.json'):
        if not (checkpoint/name).is_file():
            parser.error(f'Missing checkpoint asset: {checkpoint/name}')
    return [sys.executable, str(Path(__file__).with_name('benchmark_inference_matrix.py')),
            '--before', str(checkpoint), '--after', str(checkpoint), *rest]


if __name__ == '__main__':
    if sys.argv[1:] in (['-h'], ['--help']):
        print(__doc__+'\nRequired: --checkpoint PATH --model PATH --data JSONL --output NEW_DIR.\n'
              'Further options: see benchmark_inference_matrix.py --help.')
    else:
        raise SystemExit(subprocess.call(command(sys.argv[1:])))
