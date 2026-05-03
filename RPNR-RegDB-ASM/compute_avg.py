#!/usr/bin/env python3
"""
计算 RegDB 10 trials 测试结果的平均值和标准差。
从 test_eval/result_trial*.txt 中解析结果。
支持 Visible to Thermal 和 Thermal to Visible 两个方向。
"""

import os
import sys
import glob
import re
import numpy as np

TEST_EVAL_DIR = 'test_eval'

class Tee(object):
    """同时输出到终端和文件"""
    def __init__(self, filepath, mode='w'):
        self.file = open(filepath, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

def parse_result_file(filepath):
    """解析单个结果文件，提取 metrics。"""
    with open(filepath, 'r') as f:
        content = f.read()

    # 提取 trial 编号
    trial_match = re.search(r'Trial:\s*(\d+)', content)
    trial = int(trial_match.group(1)) if trial_match else -1

    # 匹配 "FC:   Rank-1: 89.27% | Rank-5: 93.98% | Rank-10: 95.58%| Rank-20: 97.48%| mAP: 82.00%| mINP: 67.71%"
    pattern = r'Rank-1:\s*([\d.]+)%\s*\|\s*Rank-5:\s*([\d.]+)%\s*\|\s*Rank-10:\s*([\d.]+)%\s*\|\s*Rank-20:\s*([\d.]+)%\s*\|\s*mAP:\s*([\d.]+)%\s*\|\s*mINP:\s*([\d.]+)%'
    matches = re.findall(pattern, content)

    if len(matches) == 0:
        return None
    elif len(matches) >= 1:
        v2t = [float(x) for x in matches[0]]
        t2v = [float(x) for x in matches[1]] if len(matches) > 1 else None

    return {
        'trial': trial,
        'v2t': v2t,
        't2v': t2v,
    }

def main():
    # 汇总结果同时保存到文件
    summary_file = os.path.join(TEST_EVAL_DIR, 'summary_10trials_avg.txt')
    tee = Tee(summary_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    result_files = sorted(glob.glob(os.path.join(TEST_EVAL_DIR, 'result_trial*.txt')))
    if not result_files:
        print(f"Error: No result files found in {TEST_EVAL_DIR}/")
        sys.stdout = original_stdout
        tee.close()
        return

    results = []
    for f in result_files:
        parsed = parse_result_file(f)
        if parsed and parsed['trial'] > 0:
            results.append(parsed)

    # 去重：同一 trial 可能有多个时间戳的文件，取最新的（按文件名排序后最后一个）
    trial_dict = {}
    for r in results:
        trial_dict[r['trial']] = r

    trials = sorted(trial_dict.keys())
    print(f"Found {len(trials)} valid trials: {trials}\n")

    if not trials:
        return

    metrics_names = ['Rank-1', 'Rank-5', 'Rank-10', 'Rank-20', 'mAP', 'mINP']

    # Visible to Thermal
    print("=" * 80)
    print("Visible to Thermal")
    print("=" * 80)
    v2t_matrix = np.array([trial_dict[t]['v2t'] for t in trials])
    print(f"{'Trial':<8}", end="")
    for name in metrics_names:
        print(f"{name:<12}", end="")
    print()
    for t, vals in zip(trials, v2t_matrix):
        print(f"Trial {t:<3}", end="")
        for v in vals:
            print(f"{v:<12.2f}", end="")
        print()

    print("-" * 80)
    means = v2t_matrix.mean(axis=0)
    stds = v2t_matrix.std(axis=0)
    print(f"{'Mean':<8}", end="")
    for m in means:
        print(f"{m:<12.2f}", end="")
    print()
    print(f"{'Std':<8}", end="")
    for s in stds:
        print(f"{s:<12.2f}", end="")
    print()
    print()

    # Thermal to Visible
    has_t2v = all(trial_dict[t]['t2v'] is not None for t in trials)
    if has_t2v:
        print("=" * 80)
        print("Thermal to Visible")
        print("=" * 80)
        t2v_matrix = np.array([trial_dict[t]['t2v'] for t in trials])
        print(f"{'Trial':<8}", end="")
        for name in metrics_names:
            print(f"{name:<12}", end="")
        print()
        for t, vals in zip(trials, t2v_matrix):
            print(f"Trial {t:<3}", end="")
            for v in vals:
                print(f"{v:<12.2f}", end="")
            print()

        print("-" * 80)
        means = t2v_matrix.mean(axis=0)
        stds = t2v_matrix.std(axis=0)
        print(f"{'Mean':<8}", end="")
        for m in means:
            print(f"{m:<12.2f}", end="")
        print()
        print(f"{'Std':<8}", end="")
        for s in stds:
            print(f"{s:<12.2f}", end="")
        print()
        print()

    # 简洁汇总（RegDB 通常只报 Visible to Thermal）
    print("=" * 80)
    print("Summary (Visible to Thermal) - RegDB Standard")
    print("=" * 80)
    v2t_means = v2t_matrix.mean(axis=0)
    v2t_stds = v2t_matrix.std(axis=0)
    print(f"Rank-1: {v2t_means[0]:.2f}% ± {v2t_stds[0]:.2f}%")
    print(f"Rank-5: {v2t_means[1]:.2f}% ± {v2t_stds[1]:.2f}%")
    print(f"Rank-10: {v2t_means[2]:.2f}% ± {v2t_stds[2]:.2f}%")
    print(f"Rank-20: {v2t_means[3]:.2f}% ± {v2t_stds[3]:.2f}%")
    print(f"mAP: {v2t_means[4]:.2f}% ± {v2t_stds[4]:.2f}%")
    print(f"mINP: {v2t_means[5]:.2f}% ± {v2t_stds[5]:.2f}%")
    print()
    print(f"Summary also saved to: {summary_file}")

    sys.stdout = original_stdout
    tee.close()

if __name__ == '__main__':
    main()
