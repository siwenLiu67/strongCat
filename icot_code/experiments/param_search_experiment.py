#!/usr/bin/env python3
"""
param_search_experiment.py
自动化小规模超参数组合实验，寻找 total_cost 最低的参数组合。
"""

import os
import sys
import itertools
import json
import time
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icot_code.experiments.main import evaluate_t_internal
from icot_code.data_loader import load_production_data, load_transportation_data, load_drl_hyperparameters, load_search_parameters
import icot_code.config 

# 1. 定义参数搜索空间
param_grid = {
    'GAMMA': [0.95],
    'LEARNING_RATE': [0.01],
    'ALPHA': [1.0],
    'BETA': [0.5],
    'COARSE_SEARCH_MAX_EPISODES': [10,20,30,40],
    'T_INTERNAL_COARSE_STEP': [20,30,40,50], 
    'T_INTERNAL_FINE_STEP' : [4,6,8,10], 
    'T_INTERNAL_FINE_SEARCH_RANGE ':[10, 20, 30, 40] # 精细搜索范围 (小时)'

}

# 2. 生成所有参数组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 3. 加载算例数据（可根据需要修改实例路径）
instance_id = "instance_m10_j20_s1"
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "instances", instance_id + ".json")
prod_data = load_production_data(file_path)
trans_data = load_transportation_data(file_path)
drl_params = load_drl_hyperparameters()
search_params = load_search_parameters()

# 4. 固定T_internal和max_episodes，专注于超参数影响
T_INTERNAL = 200  # 可根据实际问题调整
MAX_EPISODES = 10

results = []

for params in param_combinations:
    # 更新drl_params
    drl_params['gamma'] = params['GAMMA']
    drl_params['learning_rate'] = params['LEARNING_RATE']
    drl_params['alpha'] = params['ALPHA']
    drl_params['beta'] = params['BETA']
    drl_params['COARSE_SEARCH_MAX_EPISODES'] = params['COARSE_SEARCH_MAX_EPISODES']
    
    print(f"\n=== 测试参数: {params} ===")
    t0 = time.time()
    t_internal, total_cost, plan, c_max = evaluate_t_internal(
        T_INTERNAL,
        prod_data=prod_data,
        trans_data=trans_data,
        drl_params=drl_params,
        
        max_episodes=MAX_EPISODES
    )
    elapsed = time.time() - t0
    results.append({
        'params': params,
        'total_cost': total_cost,
        'c_max': c_max,
        'time': elapsed
    })
    print(f"参数: {params}, total_cost: {total_cost}, c_max: {c_max}, time: {elapsed:.2f}s")

# 5. 保存结果
with open("param_search_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n所有参数组合测试完成，结果已保存到 param_search_results.json")
