"""Run one paper parameter scan, sequentially on the visible GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run import experiment_definitions

ROOT = Path(__file__).resolve().parents[1]


def main():
    groups = json.loads((ROOT / 'configs/sweeps.json').read_text())['groups']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, choices=['all', *groups])
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/paper.json').read_text())
    definitions = experiment_definitions(config)
    selected = groups.values() if args.group == 'all' else [groups[args.group]]
    names = list(dict.fromkeys(name for group in selected for name in group['runs']))
    # A shared output root reuses C0/C1/C2/C3 across scans only after successful completion.
    for name in names:
        output = args.output_root / name
        receipt = output / 'runtime.json'
        protocol = output / 'protocol.json'
        if receipt.is_file() and protocol.is_file():
            status = json.loads(receipt.read_text())
            recorded = json.loads(protocol.read_text())
            if (status['exit_code'] == 0 and recorded['experiment'] == name
                    and Path(recorded['command'][recorded['command'].index('--source_path') + 1])
                    == args.data.resolve()
                    and recorded['config'] == config
                    and recorded.get('overrides') == definitions[name][1]
                    and (output / 'point_cloud/iteration_665000/point_cloud.ply').is_file()):
                print('Reusing completed run: ' + name, flush=True)
                continue
        command = [sys.executable, str(ROOT / 'scripts/run.py'), 'train',
                   '--data', str(args.data), '--output', str(output), '--experiment', name]
        if args.dry_run:
            command.append('--dry-run')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
