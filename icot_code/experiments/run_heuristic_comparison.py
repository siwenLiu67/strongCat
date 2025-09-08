#!/usr/bin/env python3
"""
运行启发式算法对比实验
在多个实例上测试三个启发式算法的性能
"""

import sys
import os
import json
import time
from typing import Dict, List
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icot_code.comparison_algorithms.heuristic_algorithms import SPTRule, EDDRule, CompositeRule
from icot_code.data_loader import load_production_data, load_transportation_data
import icot_code.config


def run_experiment_on_instance(instance_path: str, T_internal_values: List[int]) -> List[Dict]:
    """
    在单个实例上运行所有启发式算法
    
    Args:
        instance_path: 实例文件路径
        T_internal_values: 要测试的T_internal值列表
        
    Returns:
        包含所有算法结果的字典列表
    """
    # 加载实例数据
    production_data = load_production_data(instance_path)
    transportation_data = load_transportation_data(instance_path)
    orders_data = transportation_data['orders']
    
    # 初始化算法
    algorithms = [
        SPTRule(),
        EDDRule(),
        CompositeRule()
    ]
    
    results = []
    
    for T_internal in T_internal_values:
        for algorithm in algorithms:
            print(f"运行 {algorithm.name} 在 T_internal={T_internal}")
            
            start_time = time.time()
            result = algorithm.solve(production_data, orders_data, T_internal)
            end_time = time.time()
            
            # 提取关键指标
            metrics = result['metrics']
            result_info = {
                'instance': os.path.basename(instance_path),
                'algorithm': algorithm.name,
                'T_internal': T_internal,
                'makespan': metrics['makespan'],
                'total_cost': metrics['total_cost'],
                'production_cost': metrics['production_cost'],
                'transportation_cost': metrics['transportation_cost'],
                'tardiness': metrics['tardiness'],
                'feasible': metrics['feasible'],
                'computation_time': end_time - start_time,
                'schedule_length': len(result['schedule'])
            }
            
            results.append(result_info)
    
    return results


def main():
    """主实验函数"""
    print("=== 启发式算法对比实验 ===")
    
    # 定义要测试的实例
    instances = [
        "icot_code/instances/instance_m10_j20_s1.json",
        "icot_code/instances/instance_m10_j20_s2.json",
        "icot_code/instances/instance_m10_j20_s3.json"
    ]
    
    # 定义要测试的T_internal值
    T_internal_values = [50, 100, 150]
    
    all_results = []
    
    for instance_path in instances:
        if not os.path.exists(instance_path):
            print(f"警告: 实例文件 {instance_path} 不存在，跳过")
            continue
        
        print(f"\n处理实例: {os.path.basename(instance_path)}")
        instance_results = run_experiment_on_instance(instance_path, T_internal_values)
        all_results.extend(instance_results)
    
    # 保存结果到CSV文件
    if all_results:
        df = pd.DataFrame(all_results)
        output_file = "results/heuristic_comparison_results.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 打印汇总统计
        print("\n=== 汇总统计 ===")
        summary = df.groupby(['algorithm', 'T_internal']).agg({
            'makespan': ['mean', 'std'],
            'total_cost': ['mean', 'std'],
            'feasible': 'mean',
            'computation_time': 'mean'
        }).round(2)
        
        print(summary)
        
    else:
        print("没有生成任何结果")


if __name__ == "__main__":
    main()
