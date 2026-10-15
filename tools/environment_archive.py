"""校验环境 tar 的 SHA-256 并在本地恢复。"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid


def transfer(source, destination, expected_size, expected_hash):
    started = time.monotonic()
    state = {'bytes': 0}
    stopped = threading.Event()
    def report():
        elapsed = max(time.monotonic() - started, 0.001)
        count = state['bytes']
        rate = count / elapsed
        eta = f'{(expected_size-count)/rate:.0f}s' if rate else '未知'
        print(f'[环境包传输] {count/expected_size:.1%} '
              f'{count/1024**3:.2f}/{expected_size/1024**3:.2f} GiB，'
              f'平均 {rate/1024**2:.2f} MiB/s，已用 {elapsed:.0f}s，剩余约 {eta}', flush=True)
    def monitor():
        while not stopped.wait(15):
            report()
    thread = threading.Thread(target=monitor, daemon=True)
    digest = hashlib.sha256()
    print(f'[环境包传输] {source}', flush=True)
    thread.start()
    try:
        with source.open('rb') as src, destination.open('wb') as dst:
            while True:
                chunk = src.read(8 * 1024**2)
                if not chunk:
                    break
                dst.write(chunk)
                digest.update(chunk)
                state['bytes'] += len(chunk)
        if state['bytes'] != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError('环境包大小或 SHA-256 校验失败，未解压')
        report()
    finally:
        stopped.set()
        thread.join()
    print('[环境包] SHA-256 校验通过', flush=True)


def restore_environment(archive, target):
    archive, target = Path(archive), Path(target)
    manifest = json.loads(Path(str(archive) + '.json').read_text())
    digest = manifest['sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('无效的 SHA-256 清单')
    size = int(manifest['size'])
    if size <= 0:
        raise ValueError('环境包不能为空')
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ('.' + target.name + '.restore.lock')).open('w') as lock:
        print('[环境] 等待本地恢复锁', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = target / '.archive_restored.json'
        if marker.exists() and (target / 'bin/python').is_file():
            if json.loads(marker.read_text()).get('sha256') == digest:
                print('[环境] 复用同一版本已恢复环境', flush=True)
                return str(target / 'bin/python')
        available = shutil.disk_usage(target.parent).free
        required = size + int(manifest.get('unpacked_bytes', size))
        if available < required:
            raise RuntimeError(f'本地磁盘不足：至少需要 {required/1024**3:.1f} GiB，剩余 {available/1024**3:.1f} GiB')
        # 与旧版逐文件复制的锁兼容，不替换正在复制的目录。
        target.mkdir(parents=True, exist_ok=True)
        with (target / '.setup.lock').open('w') as old_lock:
            fcntl.flock(old_lock, fcntl.LOCK_EX)
            with tempfile.TemporaryDirectory(prefix='guidedgs_unpack_', dir=target.parent) as work:
                work = Path(work)
                local_archive = work / 'environment.tar'
                transfer(archive, local_archive, size, digest)
                stage = work / 'environment'
                stage.mkdir()
                print('[本地解压] 开始', flush=True)
                start = time.monotonic()
                with subprocess.Popen(['tar', '--extract', '--file', str(local_archive),
                                       '--directory', str(stage), '--no-same-owner']) as proc:
                    while True:
                        try:
                            result = proc.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            print(f'[本地解压] 已用 {time.monotonic()-start:.0f}s', flush=True)
                    if result:
                        raise RuntimeError(f'tar 解压失败，退出码 {result}')
                python = stage / 'bin/python'
                if not python.is_file():
                    raise RuntimeError('环境包缺少 bin/python')
                for path in (stage / 'bin').iterdir():
                    if path.is_file():
                        path.chmod(path.stat().st_mode | 0o100)
                (stage / '.archive_restored.json').write_text(json.dumps(manifest))
                previous = target.with_name(target.name + '.previous_' + uuid.uuid4().hex[:8])
                target.rename(previous)
                try:
                    stage.rename(target)
                except BaseException:
                    previous.rename(target)
                    raise
                # 保留有内容的旧环境；仅移除本次创建的空锁目录。
                if set(p.name for p in previous.iterdir()) == {'.setup.lock'}:
                    (previous / '.setup.lock').unlink()
                    previous.rmdir()
                else:
                    print(f'[环境] 旧目录保留在 {previous}', flush=True)
                print(f'[本地解压] 完成，用时 {time.monotonic()-start:.0f}s', flush=True)
    return str(target / 'bin/python')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--target', default='envs/guidedgs')
    args = parser.parse_args()
    restore_environment(args.archive, args.target)
