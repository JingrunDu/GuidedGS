"""Download shared datasets and environments with checksum validation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('section', choices=['data/subset_6650', 'data/full', 'environment'])
    parser.add_argument('--destination', type=Path, default=Path('downloads'))
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--environment', default='industrialgs_py310_cu121_v1',
                        help='Environment directory to download')
    args = parser.parse_args()
    if not args.environment or '/' in args.environment or args.environment in {'.', '..'}:
        parser.error('Invalid environment directory name')
    section = args.section + '/' + args.environment if args.section == 'environment' else args.section
    from modelscope_hub import HubApi
    token = args.token_file.read_text().strip() if args.token_file else None
    api = HubApi(token=token)
    repo = 'JingrunDu/converter_value'
    entries = [f for f in api.list_repo_files(repo, 'dataset')
               if f.type == 'blob' and f.path.startswith(section + '/')]
    if not entries:
        raise FileNotFoundError('No files found for selected section')
    for entry in entries:
        api.download_file(repo, 'dataset', entry.path, local_dir=args.destination,
                          expected_sha256=entry.sha256 or None)
        print('Downloaded:', entry.path, flush=True)


if __name__ == '__main__':
    main()
