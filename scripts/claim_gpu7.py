#!/usr/bin/env python3
"""Wait for physical GPU 7 on Allegro and hold memory until stopped or timed out.

Run with the project's Python environment. This is a cooperative memory hold,
not a scheduler reservation; other users can still attempt to use the GPU.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=4, help='Hold duration after acquisition (default: 4 hours).')
    parser.add_argument('--memory-fraction', type=float, default=0.95, help='Fraction of available memory to hold (default: 0.95).')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.hours <= 0 or not 0 < args.memory_fraction < 1 or args.poll_seconds <= 0:
        parser.error('hours and poll-seconds must be positive; memory-fraction must be between 0 and 1')
    if socket.gethostname().split('.')[0] != 'allegro':
        parser.error('This script is restricted to Allegro.')

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    root = Path(__file__).resolve().parents[1]
    locks = []
    for relative in ('logs/research/queue.lock', 'logs/distance-2048-v1/queue.lock'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('a')
        locks.append(handle)
        log(f'Waiting for project lock: {relative}')
        fcntl.flock(handle, fcntl.LOCK_EX)

    gpu_uuid = query('-i', '7', '--query-gpu=uuid', '--format=csv,noheader')

    def other_processes():
        rows = query('--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits')
        return [row for row in rows.splitlines()
                if row.split(',')[0].strip() == gpu_uuid
                and int(row.split(',')[1].strip()) != os.getpid()]

    log(f'Waiting for physical GPU 7 ({gpu_uuid}); no jobs will be stopped.')
    while True:
        memory, utilization = map(int, query('-i', gpu_uuid, '--query-gpu=memory.used,utilization.gpu',
                                            '--format=csv,noheader,nounits').split(','))
        if memory < 1000 and utilization < 10 and not other_processes():
            break
        time.sleep(args.poll_seconds)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    import torch
    torch.cuda.init()
    if other_processes():
        raise RuntimeError('Another job arrived during acquisition. No hold allocated; rerun to wait again.')
    free, _ = torch.cuda.mem_get_info(0)
    held = torch.empty(int(free * args.memory_fraction), dtype=torch.uint8, device='cuda:0')
    torch.cuda.synchronize()
    if other_processes():
        del held
        torch.cuda.empty_cache()
        raise RuntimeError('Another job arrived during acquisition. Hold released; rerun to wait again.')
    log(f'HOLDING GPU 7: PID {os.getpid()}, {held.numel() / 2**30:.2f} GiB, for {args.hours:g} hours. Stop with Ctrl-C or kill -TERM {os.getpid()}.')
    try:
        deadline = time.monotonic() + args.hours * 3600
        while time.monotonic() < deadline:
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    finally:
        del held
        torch.cuda.empty_cache()
        log('GPU 7 memory hold released.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Stopped; GPU memory and project locks are released on exit.')
