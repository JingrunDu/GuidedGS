"""Download and restore one dataset and a versioned environment before training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['subset_6650', 'full'], default='subset_6650')
    parser.add_argument('--environment', default='industrialgs_py310_cu121_v1')
    parser.add_argument('--downloads', type=Path, default=Path('downloads'))
    parser.add_argument('--data', type=Path, default=Path('data/perspective'))
    parser.add_argument('--env', type=Path, default=Path('envs/guidedgs'))
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.environment or '/' in args.environment or args.environment in {'.', '..'}:
        parser.error('Invalid environment directory name')
    downloads, data, environment = args.downloads.resolve(), args.data.resolve(), args.env.resolve()
    if data.exists() and any(data.iterdir()):
        parser.error('Use a new empty --data directory; existing data can be passed directly to local_run.py')
    package = downloads / 'environment' / args.environment
    tools = ROOT / 'tools'
    auth = ['--token-file', str(args.token_file.resolve())] if args.token_file else []
    commands = [
        [sys.executable, str(tools / 'download_materials.py'), 'data/' + args.dataset,
         '--destination', str(downloads), *auth],
        [sys.executable, str(tools / 'download_materials.py'), 'environment',
         '--environment', args.environment, '--destination', str(downloads), *auth],
        [sys.executable, str(tools / 'restore_delivery.py'), 'data',
         '--source', str(downloads / 'data' / args.dataset), '--target', str(data)],
        [sys.executable, str(tools / 'restore_delivery.py'), 'merge', '--source', str(package)],
        [sys.executable, str(tools / 'environment_archive.py'),
         '--archive', str(package / 'industrialgs_py310_v1.tar'), '--target', str(environment)],
        [str(environment / 'bin/python'), '-m', 'pip', 'install', '--no-index',
         '--find-links', str(package / 'wheels'), '-r', str(package / 'requirements-runtime.txt')],
    ]
    clean_env = dict(os.environ)
    clean_env.pop('PYTHONHOME', None)
    clean_env.pop('PYTHONPATH', None)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            # The invoking Python may use a separately installed download SDK.
            env = clean_env if command[0] == str(environment / 'bin/python') else None
            subprocess.run(command, check=True, env=env)
    print('Data:', data, '\nTraining Python:', environment / 'bin/python', flush=True)


if __name__ == '__main__':
    main()
