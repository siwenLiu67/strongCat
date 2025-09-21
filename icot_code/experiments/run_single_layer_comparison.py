#!/usr/bin/env python3
"""
运行单层深度强化学习算法对比实验
在多个实例上测试单层DRL算法的性能，并与启发式算法进行对比
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
from icot_code.comparison_algorithms.single_layer_drl import DQNAlgorithm, PPOAlgorithm, A2CAlgorithm
from icot_code.data_loader import load_production_data, load_transportation_data
from icot_code.models.production_env import FJSPEnv
import icot_code.config
from icot_code.models.experiment_result import ExperimentResult, save_experiment_result


def run_experiment_on_instance(instance:str, instance_path: str) -> List[Dict]:
    """
    在单个实例上运行所有算法（启发式 + 单层DRL）
    
    Args:
        instance: 实例ID
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
       # SPTRule(),
      #  EDDRule(),
       # CompositeRule(),
      #  DQNAlgorithm(),
        PPOAlgorithm(),
      #  A2CAlgorithm()
    ]
    
    results = []
    for algorithm in algorithms:
        print(f"运行 {algorithm.name}")  
            
        result = algorithm.solve(production_data, transportation_data, orders_data)
        print(f"  - 调度完成 (总成本={result['metrics']['total_cost']}).")  
            
        # 提取关键指标
        experiment_result = ExperimentResult(
        algorithm_name= algorithm.name,
        instance_id=instance,
            metrics=result['metrics'],
            extra_info={'reward_curve': result.get('extra_info', {}).get('reward_curve', [])}
            )
                    
        save_experiment_result(experiment_result, save_dir="results", filetype="csv")
        results.append(experiment_result)
    
    return results


def main():
    """主实验函数"""
    print("=== 单层深度强化学习算法对比实验 ===")
    
    # 定义要测试的实例（使用较小的实例以节省计算时间）
    instances = [
      #  "instance_m10_j20_s1",
        "instance_m10_j20_s2",
    ]
    
    
    all_results = []
    
    for instance in instances:
        instance_path = "F://research_lsw//instances//instance_m10_j20_s1.json"
        if not os.path.exists(instance_path):
            print(f"警告: 实例文件 {instance_path} 不存在，跳过")
            continue
        
        print(f"\n=== 处理实例: {instance_path} ===")
        instance_results = run_experiment_on_instance(instance, instance_path)
        all_results.extend(instance_results)
    
    # 保存结果到CSV文件
    # 保存和汇总结果
    if all_results:
        # 将结果列表（包含ExperimentResult对象）转换为字典列表
        results_dicts = [res.to_dict() for res in all_results]
        
        # 创建一个基础信息的DataFrame
        df_base = pd.DataFrame(results_dicts)[['algorithm_name', 'instance_id']]
        
        # 将metrics列展开为一个新的DataFrame
        df_metrics = pd.json_normalize([d['metrics'] for d in results_dicts])
        
        # 合并两个DataFrame
        df = pd.concat([df_base, df_metrics], axis=1)

        output_file = "results/single_layer_comparison_results.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")
        
        # 打印汇总统计
        print("\n=== 汇总统计 ===")
        # 定义要统计的指标
        agg_metrics = {
            'total_cost': ['mean', 'std'],
            'computation_time': 'mean'
        }
        # 检查 'transport_feasible' 列是否存在
        if 'transport_feasible' in df.columns:
            agg_metrics['transport_feasible'] = 'mean'

        summary = df.groupby(['algorithm_name', 'instance_id']).agg(agg_metrics).round(2)
        
        print(summary)
        
        # 按算法类型分组统计
        print("\n=== 按算法类型统计 ===")
        heuristic_algorithms = ['SPT_Rule', 'EDD_Rule', 'CompositeRule']
        drl_algorithms = ['DQN_Algorithm', 'PPO_Algorithm', 'A2C_Algorithm']
        
        # 使用 'algorithm_name' 列进行筛选
        heuristic_df = df[df['algorithm_name'].isin(heuristic_algorithms)]
        drl_df = df[df['algorithm_name'].isin(drl_algorithms)]
        
        if not heuristic_df.empty:
            print("启发式算法统计:")
            print(heuristic_df.groupby('algorithm_name').agg({
                'total_cost': ['mean', 'std'],
                'computation_time': 'mean'
            }).round(2))
        
        if not drl_df.empty:
            print("\n单层DRL算法统计:")
            print(drl_df.groupby('algorithm_name').agg({
                'total_cost': ['mean', 'std'],
                'computation_time': 'mean'
            }).round(2))
        
    else:
        print("没有生成任何结果")


if __name__ == "__main__":
    main()
