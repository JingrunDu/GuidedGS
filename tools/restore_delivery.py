"""Verify and extract local ModelScope delivery downloads without network access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge(record):
    spec = json.loads(record.read_text())
    target = record.parent / spec['original']
    if target.exists():
        if sha(target) != spec['sha256']:
            raise ValueError('Existing target differs: %s' % target)
        return
    partial = target.with_name(target.name + '.partial')
    with partial.open('wb') as output:
        for entry in spec['parts']:
            source = record.parent / entry['file']
            if source.stat().st_size != entry['size'] or sha(source) != entry['sha256']:
                raise ValueError('Part checksum mismatch: %s' % source)
            with source.open('rb') as handle:
                shutil.copyfileobj(handle, output, length=8 * 1024**2)
    if partial.stat().st_size != spec['original_size'] or sha(partial) != spec['sha256']:
        raise ValueError('Merged file checksum mismatch')
    partial.replace(target)
    print('Merged:', target, flush=True)


def extract_data(source, target):
    spec = json.loads((source / 'dataset.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError('Use a new empty extraction directory')
    for entry in [spec['metadata']] + spec['shards']:
        archive = source / entry['file']
        if sha(archive) != entry['sha256']:
            raise ValueError('Archive checksum mismatch: %s' % archive)
        with tarfile.open(archive) as tar:
            for member in tar:
                destination = (target / member.name).resolve()
                if target.resolve() not in destination.parents or not member.isfile():
                    raise ValueError('Unsafe archive member: %s' % member.name)
                if destination.exists():
                    raise ValueError('Duplicate archive member: %s' % member.name)
                tar.extract(member, target)
        print('Extracted:', archive.name, flush=True)
    shutil.copyfile(source / 'images.txt', target / 'images.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['merge', 'data'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path)
    parser.add_argument('--remove-parts', action='store_true', help='Remove verified pieces after merging')
    args = parser.parse_args()
    if args.mode == 'merge':
        records = [args.source] if args.source.is_file() else sorted(args.source.rglob('*.parts.json'))
        if not records:
            raise FileNotFoundError('No parts manifests found')
        for record in records:
            merge(record)
            if args.remove_parts:
                for entry in json.loads(record.read_text())['parts']:
                    (record.parent / entry['file']).unlink()
    else:
        if args.target is None:
            parser.error('data requires --target')
        extract_data(args.source, args.target)


if __name__ == '__main__':
    main()
