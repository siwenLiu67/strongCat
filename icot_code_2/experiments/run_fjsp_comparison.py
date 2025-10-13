"""
主实验运行脚本
Main Experiment Runner Script
"""
import os
import sys
import time
import glob
import csv
import multiprocessing
from tqdm import tqdm
import traceback

# --- 路径设置和模块导入 ---
# 确保项目根目录在 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from models.fjsp_env import FjspEnv
from comparison_algorithms.single_layer_drl.dqn import DQN_Agent
from comparison_algorithms.single_layer_drl.a2c import A2C_Agent
from comparison_algorithms.single_layer_drl.ppo import PPO_Agent
from comparison_algorithms.metaheuristic_algorithms.ga import GA_Agent
from comparison_algorithms.metaheuristic_algorithms.sa import SA_Agent
from comparison_algorithms.metaheuristic_algorithms.pso import PSO_Agent

def get_all_agents(env):
    """
    获取所有已定义的算法智能体实例。
    Dynamically creates and returns instances of all defined agents.
    """
    return {
        "DQN": DQN_Agent(env, config.DQN_CONFIG),
      #  "GA": GA_Agent(env, config.GA_CONFIG),
      #  "SA": SA_Agent(env, config.SA_CONFIG),
      #  "PSO": PSO_Agent(env, config.PSO_CONFIG),
      #  "A2C": A2C_Agent(env, config.A2C_CONFIG),
      #  "PPO": PPO_Agent(env, config.PPO_CONFIG)
    }

def run_experiment_on_instance(instance_path):
    """
    对单个实例运行所有算法，并返回结果列表。
    """
    instance_name = os.path.basename(instance_path)
    print(f"--- Processing Instance: {instance_name} ---")
    
    try:
        env = FjspEnv(instance_path=instance_path)
        agents = get_all_agents(env)
        instance_results = []

        for agent_name, agent in agents.items():
            print(f"  [{instance_name}] Running {agent_name}...")
            start_time = time.time()
            
            solution, results = agent.solve()
            execution_time = time.time() - start_time
            
            row = {
                "instance": instance_name,
                "algorithm": agent_name,
                "execution_time_s": round(execution_time, 2)
            }
            if results:
                row.update(results)
            
            instance_results.append(row)
        
        return instance_results

    except Exception as e:
        print(f"FATAL ERROR processing instance {instance_name}: {e}")
        traceback.print_exc()
        return []

def print_instance_results(results):
    """
    以表格形式打印单个实例的结果。
    """
    if not results:
        return
    
    instance_name = results[0]['instance']
    print(f"\n--- Results for Instance: {instance_name} ---")
    
    # 打印表头
    headers = ["Algorithm", "Objective", "Makespan", "Energy", "Carbon", "Time (s)"]
    print(f"{headers[0]:<10} | {headers[1]:>10} | {headers[2]:>10} | {headers[3]:>10} | {headers[4]:>10} | {headers[5]:>10}")
    print("-" * 70)
    
    # 打印每一行数据
    for res in results:
        if res.get('objective_value') is not None:
            print(f"{res['algorithm']:<10} | {res['objective_value']:>10.2f} | {res['makespan']:>10.2f} | "
                  f"{res['total_energy']:>10.2f} | {res['total_carbon']:>10.2f} | {res['execution_time_s']:>10.2f}")
    print("-" * 70)


def main():
    """
    主函数，自动化整个实验流程。
    """
    print("--- Starting Batch Experiment (Sequential) ---")
    
    instances_dir = os.path.join(config.PROJECT_ROOT, 'instances')
    instance_paths = sorted(glob.glob(os.path.join(instances_dir, config.INSTANCE_GLOB_PATTERN)))
    
    if not instance_paths:
        print(f"Error: No instance files found matching '{config.INSTANCE_GLOB_PATTERN}' in '{instances_dir}'.")
        return
        
    print(f"Found {len(instance_paths)} instances to process.")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_csv_path = os.path.join(config.RESULTS_DIR, config.RESULTS_CSV_FILENAME)
    
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=config.CSV_FIELDNAMES)
        writer.writeheader()
        
        for i, instance_path in enumerate(instance_paths):
            print(f"\n[{i+1}/{len(instance_paths)}]=============================================")
            instance_results = run_experiment_on_instance(instance_path)
            
            if instance_results:
                valid_results = [res for res in instance_results if res.get('objective_value') is not None]
                if valid_results:
                    writer.writerows(valid_results)
                    print_instance_results(valid_results)

    print(f"\n--- Batch Experiment Finished ---")
    print(f"All results have been saved to: {results_csv_path}")

if __name__ == "__main__":
    main()
