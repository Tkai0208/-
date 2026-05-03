#!/usr/bin/env python3
"""
自动汇总 RegDB 10-trial 评估结果
读取 logs/regdb_eval/{trial}/test_log.txt，计算均值和标准差
"""

import os
import re
import numpy as np

logs_dir = './logs/regdb_eval'

# 数据结构：存储所有 trial 的指标
results_v2t = []  # Visible to Thermal
results_t2v = []  # Thermal to Visible

metrics_names = ['Rank-1', 'Rank-5', 'Rank-10', 'Rank-20', 'mAP', 'mINP']

print("=" * 70)
print("RegDB 10-Trial Evaluation Summary")
print("=" * 70)

for trial in range(1, 11):
    log_file = os.path.join(logs_dir, str(trial), 'test_log.txt')
    if not os.path.exists(log_file):
        print(f"[Trial {trial:2d}] NOT FOUND, skipping...")
        continue

    with open(log_file, 'r') as f:
        content = f.read()

    # 正则提取: "FC:   Rank-1: 87.23% | Rank-5: 91.26% | ..."
    # 注意: 日志中部分 pipe 前可能没有空格 (如 "93.50%| Rank-20")
    pattern = r"Rank-1:\s+([\d.]+)%\s*\|\s*Rank-5:\s+([\d.]+)%\s*\|\s*Rank-10:\s+([\d.]+)%\s*\|\s*Rank-20:\s+([\d.]+)%\s*\|\s*mAP:\s+([\d.]+)%\s*\|\s*mINP:\s+([\d.]+)%"
    matches = re.findall(pattern, content)

    if len(matches) >= 2:
        v2t = [float(x) for x in matches[0]]   # 第一个 match = visible to thermal
        t2v = [float(x) for x in matches[1]]   # 第二个 match = thermal to visible
        results_v2t.append(v2t)
        results_t2v.append(t2v)
        print(f"[Trial {trial:2d}] V→T Rank-1: {v2t[0]:.2f}% | mAP: {v2t[4]:.2f}%  |  T→V Rank-1: {t2v[0]:.2f}% | mAP: {t2v[4]:.2f}%")
    elif len(matches) == 1:
        print(f"[Trial {trial:2d}] Only one direction found, skipping...")
    else:
        print(f"[Trial {trial:2d}] No valid metrics found, skipping...")

if len(results_v2t) == 0:
    print("\nNo valid trial results found!")
    exit(1)

results_v2t = np.array(results_v2t)
results_t2v = np.array(results_t2v)

# 计算均值和标准差
v2t_mean = results_v2t.mean(axis=0)
v2t_std = results_v2t.std(axis=0)
t2v_mean = results_t2v.mean(axis=0)
t2v_std = results_t2v.std(axis=0)

# 打印汇总
print("\n" + "=" * 70)
print(f"Summary: {len(results_v2t)} valid trials")
print("=" * 70)

print("\n>> Visible to Thermal")
print("-" * 50)
print(f"{'Metric':<12} {'Mean':<12} {'Std':<12}")
print("-" * 50)
for i, name in enumerate(metrics_names):
    print(f"{name:<12} {v2t_mean[i]:>8.2f}%    {v2t_std[i]:>8.2f}%")

print("\n>> Thermal to Visible")
print("-" * 50)
print(f"{'Metric':<12} {'Mean':<12} {'Std':<12}")
print("-" * 50)
for i, name in enumerate(metrics_names):
    print(f"{name:<12} {t2v_mean[i]:>8.2f}%    {t2v_std[i]:>8.2f}%")

# 保存到文件
output_file = os.path.join(logs_dir, 'summary.txt')
with open(output_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("RegDB 10-Trial Evaluation Summary\n")
    f.write("=" * 60 + "\n")
    f.write(f"Valid trials: {len(results_v2t)}\n\n")

    f.write(">>> Visible to Thermal <<<\n")
    f.write("-" * 40 + "\n")
    for i, name in enumerate(metrics_names):
        f.write(f"{name:<12}: {v2t_mean[i]:.2f}% ± {v2t_std[i]:.2f}%\n")

    f.write("\n>>> Thermal to Visible <<<\n")
    f.write("-" * 40 + "\n")
    for i, name in enumerate(metrics_names):
        f.write(f"{name:<12}: {t2v_mean[i]:.2f}% ± {t2v_std[i]:.2f}%\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("Individual Trial Results:\n")
    f.write("-" * 60 + "\n")
    for idx, trial_num in enumerate(range(1, 11)):
        if idx < len(results_v2t):
            f.write(f"Trial {trial_num:2d}: V→T Rank-1={results_v2t[idx][0]:.2f}% mAP={results_v2t[idx][4]:.2f}% | "
                   f"T→V Rank-1={results_t2v[idx][0]:.2f}% mAP={results_t2v[idx][4]:.2f}%\n")

print(f"\n[Saved] Summary written to: {output_file}")
