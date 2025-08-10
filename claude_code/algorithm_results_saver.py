import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pickle

def save_algorithm_results_csv(
    algo_name: str,
    instance_id: str, 
    seed: int,
    config: Any,
    stats: Dict,
    env: Any,
    total_time: float,
    additional_metrics: Optional[Dict] = None
):
    """
    规范化保存算法结果到CSV文件
    
    Args:
        algo_name: 算法名称 (如 "RuleDQN", "HRL+GAT", "Option-Critic")
        instance_id: 算例ID (如 "Small-01", "Medium-02") 
        seed: 随机种子
        config: 配置对象
        stats: 统计数据字典
        env: 环境对象
        total_time: 总运行时间(秒)
        additional_metrics: 额外的算法特定指标
    """
    output_file = Path("results") / "algorithm_comparison.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    # 基础运行信息
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
    
    # 计算运行效率
    episode_lengths = stats.get('episode_lengths', [])
    episode_rewards = stats.get('episode_rewards', [])
    makespans = stats.get('makespans', [])
    
    num_episodes = len(episode_rewards) if episode_rewards else 0
    num_steps = int(sum(episode_lengths)) if episode_lengths else 1
    time_per_step_ms = (total_time / num_steps * 1000.0) if num_steps > 0 else 0.0
    time_per_episode_s = (total_time / num_episodes) if num_episodes > 0 else 0.0
    
    # 获取作业列表
    jobs_list = getattr(env, 'jobs', [])
    if hasattr(env, 'case') and hasattr(env.case, 'jobs'):
        jobs_list = env.case.jobs
    
    # 1) 总延迟计算
    total_tardiness = 0.0
    for job in jobs_list:
        due = float(getattr(job, 'due_date', 0.0))
        completed = float(getattr(job, 'completed_time', getattr(job, 'completion_time', 0.0)))
        if completed > 0:  # 只计算已完成的作业
            total_tardiness += max(0.0, completed - due)
    
    # 2) 加权短缺计算
    weighted_shortage = 0.0
    distributors = getattr(env, 'distributors', [])
    if hasattr(env, 'case') and hasattr(env.case, 'distributors'):
        distributors = env.case.distributors
    
    dispatched_jobs = getattr(env, 'dispatched_jobs', [])
    
    for dist in distributors:
        dist_id = getattr(dist, 'distributor_id', getattr(dist, 'id', None))
        required = int(getattr(dist, 'required_jobs', getattr(dist, 'demand', 0)))
        weight = float(getattr(dist, 'weight', 1.0))
        
        delivered = sum(1 for j in dispatched_jobs
                       if getattr(j, 'distributor_id', getattr(j, 'assigned_distributor', None)) == dist_id)
        shortage = max(0, required - delivered)
        weighted_shortage += weight * shortage
    
    # 3) 目标函数值
    objective_sum = total_tardiness + weighted_shortage
    
    # 4) 性能指标
    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    best_reward = float(np.max(episode_rewards)) if episode_rewards else 0.0
    
    avg_makespan = float(np.mean(makespans)) if makespans else 0.0
    final_makespan = float(makespans[-1]) if makespans else getattr(env, 'current_time', getattr(env, 't', 0.0))
    
    # 5) 调度质量指标
    if jobs_list:
        # 按时完成率
        on_time_jobs = sum(1 for j in jobs_list 
                          if float(getattr(j, 'completed_time', getattr(j, 'completion_time', 0.0))) <= 
                             float(getattr(j, 'due_date', 0.0)))
        on_time_rate = on_time_jobs / len(jobs_list)
        
        # 平均流程时间
        flow_times = []
        for job in jobs_list:
            arrival = float(getattr(job, 'arrival_time', 0.0))
            completed = float(getattr(job, 'completed_time', getattr(job, 'completion_time', 0.0)))
            if completed > 0:
                flow_times.append(completed - arrival)
        avg_flow_time = float(np.mean(flow_times)) if flow_times else 0.0
        
        # 平均等待时间
        waiting_times = []
        for job in jobs_list:
            processing_time = sum(getattr(op, 'processing_time', 0) for op in getattr(job, 'operations', []))
            arrival = float(getattr(job, 'arrival_time', 0.0))
            completed = float(getattr(job, 'completed_time', getattr(job, 'completion_time', 0.0)))
            if completed > 0 and processing_time > 0:
                waiting_times.append(max(0, completed - arrival - processing_time))
        avg_waiting_time = float(np.mean(waiting_times)) if waiting_times else 0.0
    else:
        on_time_rate = 0.0
        avg_flow_time = 0.0
        avg_waiting_time = 0.0
    
    # 6) 资源利用率
    machines = getattr(env, 'machines', [])
    if hasattr(env, 'case') and hasattr(env.case, 'machines'):
        machines = env.case.machines
    
    if machines and final_makespan > 0:
        total_busy_time = sum(getattr(m, 'total_busy_time', getattr(m, 'busy_time', 0)) for m in machines)
        machine_utilization = total_busy_time / (len(machines) * final_makespan)
    else:
        machine_utilization = 0.0
    
    # 7) 需求覆盖率
    cover_vals = []
    for dist in distributors:
        required = int(getattr(dist, 'required_jobs', getattr(dist, 'demand', 0)))
        if required > 0:
            dist_id = getattr(dist, 'distributor_id', getattr(dist, 'id', None))
            delivered = sum(1 for j in dispatched_jobs
                           if getattr(j, 'distributor_id', getattr(j, 'assigned_distributor', None)) == dist_id)
            cover_vals.append(min(1.0, delivered / required))
    demand_coverage = float(np.mean(cover_vals)) if cover_vals else 0.0
    
    # 构建基础行数据
    row = {
        # 运行标识
        "run_id": run_id,
        "algo_name": str(algo_name),
        "instance_id": str(instance_id),
        "seed": int(seed),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        
        # 问题规模
        "num_jobs": len(jobs_list),
        "num_machines": len(machines),
        "num_distributors": len(distributors),
        "num_episodes": num_episodes,
        
        # 主要目标指标
        "total_tardiness": float(total_tardiness),
        "weighted_shortage": float(weighted_shortage),
        "objective_value": float(objective_sum),
        
        # 调度性能
        "final_makespan": float(final_makespan),
        "avg_makespan": float(avg_makespan),
        "on_time_rate": float(on_time_rate),
        "avg_flow_time": float(avg_flow_time),
        "avg_waiting_time": float(avg_waiting_time),
        "machine_utilization": float(machine_utilization),
        "demand_coverage": float(demand_coverage),
        
        # 学习性能
        "avg_reward": float(avg_reward),
        "std_reward": float(std_reward),
        "best_reward": float(best_reward),
        "final_reward": float(episode_rewards[-1]) if episode_rewards else 0.0,
        
        # 计算效率
        "total_time_s": float(total_time),
        "time_per_episode_s": float(time_per_episode_s),
        "time_per_step_ms": float(time_per_step_ms),
        "total_steps": int(num_steps),
        
        # 配置参数
        "learning_rate": float(getattr(config, 'learning_rate', 0.0)),
        "batch_size": int(getattr(config, 'batch_size', 0)),
        "epsilon_start": float(getattr(config, 'epsilon_start', 0.0)),
        "epsilon_end": float(getattr(config, 'epsilon_end', 0.0)),
    }
    
    # 添加算法特定指标
    if additional_metrics:
        for key, value in additional_metrics.items():
            if isinstance(value, (int, float, bool)):
                row[f"custom_{key}"] = float(value)
            else:
                row[f"custom_{key}"] = str(value)
    
    # 写入CSV文件
    try:
        file_exists = output_file.exists()
        with open(output_file, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"结果已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        return False


def save_convergence_data(
    algo_name: str,
    instance_id: str,
    seed: int,
    stats: Dict,
    config: Any = None,
    additional_series: Optional[Dict[str, List]] = None
):
    """
    保存收敛图绘制所需的数据
    
    Args:
        algo_name: 算法名称
        instance_id: 算例ID
        seed: 随机种子
        stats: 统计数据字典，应包含episode_rewards等
        config: 配置对象
        additional_series: 额外的数据序列，如losses, epsilons等
    """
    # 创建保存目录
    convergence_dir = Path("results") / "convergence_data"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取主要的收敛数据
    convergence_data = {
        'metadata': {
            'algo_name': str(algo_name),
            'instance_id': str(instance_id),
            'seed': int(seed),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'num_jobs': getattr(config, 'num_jobs', 0) if config else 0,
                'num_machines': getattr(config, 'num_machines', 0) if config else 0,
                'num_distributors': getattr(config, 'num_distributors', 0) if config else 0,
                'learning_rate': getattr(config, 'learning_rate', 0.0) if config else 0.0,
                'batch_size': getattr(config, 'batch_size', 0) if config else 0,
                'epsilon_start': getattr(config, 'epsilon_start', 0.0) if config else 0.0,
                'epsilon_end': getattr(config, 'epsilon_end', 0.0) if config else 0.0,
            }
        },
        'series_data': {
            'episodes': list(range(1, len(stats.get('episode_rewards', [])) + 1)),
            'episode_rewards': stats.get('episode_rewards', []),
            'episode_lengths': stats.get('episode_lengths', []),
            'makespans': stats.get('makespans', []),
            'running_times': stats.get('running_times', []),
        }
    }
    
    # 计算滑动平均
    episode_rewards = stats.get('episode_rewards', [])
    if episode_rewards:
        convergence_data['series_data']['reward_moving_avg'] = [
            np.mean(episode_rewards[max(0, i-10):i+1]) for i in range(len(episode_rewards))
        ]
 
    # 计算makespan的滑动平均和趋势
    makespans = stats.get('makespans', [])
    if makespans:
        convergence_data['series_data']['makespan_moving_avg'] = [
            np.mean(makespans[max(0, i-10):i+1]) for i in range(len(makespans))
        ]

    # 保存为JSON和pickle两种格式
    filename_base = f"{algo_name}_{instance_id}_seed{seed}"
    
    # JSON格式（便于其他工具读取）
    json_file = convergence_dir / f"{filename_base}_convergence.json"
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(convergence_data, f, indent=2, ensure_ascii=False)
        print(f"收敛数据已保存到: {json_file}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
    
    # Pickle格式（完整数据）
    pkl_file = convergence_dir / f"{filename_base}_convergence.pkl"
    try:
        with open(pkl_file, 'wb') as f:
            pickle.dump(convergence_data, f)
        print(f"收敛数据已保存到: {pkl_file}")
    except Exception as e:
        print(f"保存Pickle文件时出错: {e}")
    
    return convergence_data







def generate_instance_id(config: Any) -> str:
    """根据配置生成标准化的算例ID"""
    num_jobs = getattr(config, 'num_jobs', 0)
    num_machines = getattr(config, 'num_machines', 0)
    num_distributors = getattr(config, 'num_distributors', 0)
    
    if num_jobs <= 10:
        size = "Small"
    elif num_jobs <= 20:
        size = "Medium"
    else:
        size = "Large"
    
    return f"{size}-{num_jobs}J{num_machines}M{num_distributors}D"


def set_random_seed(seed: int):
    """设置所有随机种子"""
    import random
    import numpy as np
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    random.seed(seed)
    np.random.seed(seed)


# 使用示例函数
def example_usage():
    """使用示例"""
    # 模拟统计数据
    stats = {
        'episode_rewards': [10 + i + np.random.normal(0, 5) for i in range(100)],
        'episode_lengths': [50 + np.random.randint(-10, 10) for _ in range(100)],
        'makespans': [100 - i*0.5 + np.random.normal(0, 3) for i in range(100)],
        'running_times': [i * 0.1 for i in range(100)]
    }
    
    # 额外数据序列
    additional_series = {
        'losses': [100 - i + np.random.normal(0, 2) for i in range(100)],
        'epsilon': [1.0 - i/100 for i in range(100)]
    }
    
    # 保存收敛数据
    convergence_data = save_convergence_data(
        algo_name="Example_DQN",
        instance_id="Small-6J4M2D", 
        seed=42,
        stats=stats,
        additional_series=additional_series
    )
    



if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    