"""Portable entry points for the fixed GuidedGS paper protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_camera_names(path):
    names = []
    with path.open('rb') as handle:
        count, = struct.unpack('<Q', handle.read(8))
        for _ in range(count):
            handle.read(64)
            name = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise ValueError('Truncated COLMAP camera model')
                if char == b'\0':
                    break
                name.extend(char)
            names.append(name.decode())
            points, = struct.unpack('<Q', handle.read(8))
            handle.seek(points * 24, 1)
    return names


def check_data(data, config):
    sets = {}
    for part in ['train', 'test']:
        names = (ROOT / 'configs/splits' / (part + '.txt')).read_text().splitlines()
        if len(names) != config[part + '_images'] or len(set(names)) != len(names):
            raise ValueError('Invalid ' + part + ' split')
        sets[part] = set(names)
    if sets['train'] & sets['test']:
        raise ValueError('Training/test splits overlap')
    names = read_camera_names(data / 'sparse/0/images.bin')
    if len(names) != config['num_images'] or set(names) != sets['train'] | sets['test']:
        raise ValueError('Use the independent 6,650-image dataset with its original camera model')
    for name in names:
        for path in [data / 'images' / name, data / 'mask_sam3' / Path(name).with_suffix('.png')]:
            if not path.is_file():
                raise FileNotFoundError(path)
    cloud = data / 'sparse/0/points3D.ply'
    if hashlib.sha256(cloud.read_bytes()).hexdigest() != config['initial_cloud_sha256']:
        raise ValueError('Initial point cloud checksum differs from the paper protocol')
    if not (data / 'sparse/0/cameras.bin').is_file():
        raise FileNotFoundError('Missing cameras.bin')
    print('Validated: 6,650 images, masks, initial cloud; train 5,985 / test 665', flush=True)


def training_command(data, output, config, experiment, overrides=None):
    params = dict(config['parameters'])
    for name in config['ablations'][experiment]:
        params.update(config['modules'][name])
    params.update(overrides or {})
    command = [sys.executable, '-u', str(ROOT / 'train.py'),
               '--source_path', str(data), '--model_path', str(output),
               '--test_file', str(ROOT / 'configs/splits/test.txt')]
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        else:
            command += ['--' + key, *map(str, value if isinstance(value, list) else [value])]
    return command


def experiment_definitions(config):
    definitions = {name: (name, {}) for name in config['ablations']}
    groups = json.loads((ROOT / 'configs/sweeps.json').read_text())['groups']
    for group in groups.values():
        for value, name in zip(group['values'], group['runs']):
            if name not in config['ablations']:
                definition = (group['experiment'], {group['parameter']: value})
                if name in definitions and definitions[name] != definition:
                    raise ValueError('Conflicting scan configuration: ' + name)
                definitions[name] = definition
    return definitions


def main():
    config = json.loads((ROOT / 'configs/paper.json').read_text())
    definitions = experiment_definitions(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'train', 'render', 'evaluate'])
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Training model output directory')
    parser.add_argument('--experiment', choices=sorted(definitions), default='C7')
    parser.add_argument('--iteration', type=int, default=665000)
    parser.add_argument('--views', type=int, default=6, help='Number of evenly spaced render views')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data = args.data.resolve()
    check_data(data, config)
    if args.mode == 'check':
        return 0
    if args.output is None:
        parser.error('--output is required')
    output = args.output.resolve()
    if args.mode == 'train':
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Choose an empty output directory')
        experiment, overrides = definitions[args.experiment]
        command = training_command(data, output, config, experiment, overrides)
    else:
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate.py'),
                   '--repo-dir', str(ROOT), '--repo-type', 'fastgs',
                   '--source-path', str(data), '--model-path', str(output),
                   '--split-file', str(ROOT / 'configs/splits/test.txt'),
                   '--iteration', str(args.iteration), '--mult', '0.6',
                   '--mask-mode', 'sam3', '--mask-type', 'hard', '--soft-mask-sigma', '0']
        if args.mode == 'render':
            if not 1 <= args.views <= config['test_images']:
                parser.error('--views must be between 1 and 665')
            indices = [i * (config['test_images'] - 1) // max(args.views - 1, 1) for i in range(args.views)]
            command += ['--skip-metrics', '--view-indices', ','.join(map(str, indices)),
                        '--save-renders-dir', str(output / 'visualization/render'),
                        '--save-gt-dir', str(output / 'visualization/gt'),
                        '--output-json', str(output / 'visualization/index.json')]
        else:
            command += ['--output-json', str(output / 'metrics.json'),
                        '--per-view-json', str(output / 'metrics_per_view.json')]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    if args.mode != 'train':
        return subprocess.run(command, cwd=ROOT, env=env).returncode
    output.mkdir(parents=True, exist_ok=True)
    (output / 'protocol.json').write_text(json.dumps(dict(experiment=args.experiment,
        config=config, overrides=overrides, command=command), indent=2))
    started = time.monotonic()
    with (output / 'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        code = process.wait()
    (output / 'runtime.json').write_text(json.dumps(dict(exit_code=code,
        process_seconds=time.monotonic() - started), indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
