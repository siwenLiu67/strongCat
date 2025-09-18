#!/usr/bin/env python3
"""
param_search_tinternal.py
自动化测试 T_internal 搜索空间参数对 total_cost 的影响，参考 main.py 实现。
"""
import os
import sys
import itertools
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from icot_code.experiments.main import main as main_experiment
import icot_code.config as config

# 1. 定义参数搜索空间（可根据需要扩展）
param_grid = {
    'T_INTERNAL_COARSE_STEP': [50],
    'T_INTERNAL_FINE_STEP': [5],
    'T_INTERNAL_FINE_SEARCH_RANGE': [30],
    'COARSE_SEARCH_MAX_EPISODES': [30],  # 新增：粗略搜索轮次
}

# 2. 生成所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []

# 3. 遍历参数组合，动态修改 config.py 并运行 main.py 流程
for params in param_combinations:
    # 修改 config.py 参数
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        for k, v in params.items():
            if line.strip().startswith(f'{k} ='):
                line = f'{k} = {v}  # 自动调参\n'
        new_lines.append(line)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    # 重新加载 config
    import importlib
    importlib.reload(config)
    print(f"\n=== 测试参数: {params} ===")
    t0 = time.time()
    # 运行 main.py 的主流程（直接调用 main_experiment）
    try:
        main_experiment()
        # 读取 results/ 目录下最新的结果文件，提取 total_cost
        result_file = sorted([f for f in os.listdir('results') if f.endswith('.csv')], key=lambda x: os.path.getmtime(os.path.join('results', x)), reverse=True)[0]
        with open(os.path.join('results', result_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 假设 total_cost 在 metrics 字段或首行
            for line in lines:
                if 'total_cost' in line:
                    try:
                        total_cost = float(line.strip().split(',')[1])
                        break
                    except:
                        total_cost = None
            else:
                total_cost = None
    except Exception as e:
        print(f"运行出错: {e}")
        total_cost = None
    elapsed = time.time() - t0
    results.append({
        'params': params,
        'total_cost': total_cost,
        'time': elapsed
    })
    print(f"参数: {params}, total_cost: {total_cost}, time: {elapsed:.2f}s")

# 4. 保存结果
with open("param_search_tinternal_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n所有 T_internal 搜索参数组合测试完成，结果已保存到 param_search_tinternal_results.json")
