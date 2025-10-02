#!/usr/bin/env python3
# monitor_progress.py

import os
import time
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')
TOTAL_TASKS = 1767
VALID_SIZE_THRESHOLD = 20  # 文件大小 >20字节才算合法

def count_valid_outputs(output_dir):
    count = 0
    if not os.path.isdir(output_dir):
        return 0
    for name in os.listdir(output_dir):
        pdb_dir = os.path.join(output_dir, name)
        if not os.path.isdir(pdb_dir):
            continue
        mol_csv = os.path.join(pdb_dir, "molecules.csv")
        smiles = os.path.join(pdb_dir, "final_molecule.smiles")
        if all(os.path.exists(f) and os.path.getsize(f) > VALID_SIZE_THRESHOLD
               for f in (mol_csv, smiles)):
            count += 1
    return count

def print_progress_bar(current, total, length=50):
    filled = int(length * current / total)
    bar = '█' * filled + '-' * (length - filled)
    percent = f"{100 * current / total:6.2f}%"
    print(f"\r[{bar}] {percent} ({current}/{total})", end='', flush=True)

def main():
    parser = argparse.ArgumentParser(description="实时监测输出文件夹进度")
    parser.add_argument("output_dir", help="CadreAI 输出目录")
    parser.add_argument("--interval", type=int, default=1,
                        help="刷新间隔，单位秒（默认5s）")
    args = parser.parse_args()

    while True:
        valid = count_valid_outputs(args.output_dir)
        print_progress_bar(valid, TOTAL_TASKS)
        if valid >= TOTAL_TASKS:
            break
        time.sleep(args.interval)
    print("\n所有任务已完成！")

if __name__ == "__main__":
    main()