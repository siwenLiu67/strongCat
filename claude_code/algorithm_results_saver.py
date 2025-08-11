import csv
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

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
    统一的算法结果保存函数
    自动识别强化学习或启发式算法并适配处理
    保证所有字段都在表头，避免字段数不一致
    """
    output_file = Path("results") / "algorithm_comparison.csv"
    output_file.parent.mkdir(exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{seed}"
    metrics = _extract_unified_metrics(stats, env, config, total_time)
    # 构建CSV行
    row = {
        "run_id": run_id,
        "algo_name": str(algo_name),
        "instance_id": str(instance_id),
        "seed": int(seed),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "num_jobs": metrics['num_jobs'],
        "num_machines": metrics['num_machines'],
        "num_distributors": metrics['num_distributors'],
        "num_episodes": metrics['num_episodes'],
        "total_tardiness": metrics['total_tardiness'],
        "weighted_shortage": metrics['weighted_shortage'],
        "objective_value": metrics['objective_value'],
        "final_makespan": metrics['final_makespan'],
        "avg_makespan": metrics['avg_makespan'],
        "on_time_rate": metrics['on_time_rate'],
        "avg_flow_time": metrics['avg_flow_time'],
        "avg_waiting_time": metrics['avg_waiting_time'],
        "machine_utilization": metrics['machine_utilization'],
        "demand_coverage": metrics['demand_coverage'],
        "avg_reward": metrics['avg_reward'],
        "std_reward": metrics['std_reward'],
        "best_reward": metrics['best_reward'],
        "final_reward": metrics['final_reward'],
        "total_time_s": float(total_time),
        "time_per_episode_s": metrics['time_per_episode_s'],
        "time_per_step_ms": metrics['time_per_step_ms'],
        "total_steps": metrics['total_steps'],
        "learning_rate": float(getattr(config, 'learning_rate', 0.0)),
        "batch_size": int(getattr(config, 'batch_size', 0)),
        "epsilon_start": float(getattr(config, 'epsilon_start', 0.0)),
        "epsilon_end": float(getattr(config, 'epsilon_end', 0.0)),
        "algorithm_type": metrics['algorithm_type'],
        "episode_rewards_series": ",".join([f"{r:.4f}" for r in stats.get('episode_rewards', [])]),

    }
    # 额外指标统一加 custom_ 前缀
    custom_metrics = {}
    if additional_metrics:
        for key, value in additional_metrics.items():
            custom_metrics[f"custom_{key}"] = float(value) if isinstance(value, (int, float, bool)) else str(value)
    # 合并所有字段，保证表头一致
    row.update(custom_metrics)
    # 读取已有表头，合并所有可能字段
    file_exists = output_file.exists()
    fieldnames = list(row.keys())
    if file_exists:
        with open(output_file, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames) if reader.fieldnames else []
            # 合并已有字段和新字段
            for k in old_fields:
                if k not in fieldnames:
                    fieldnames.append(k)
            for k in fieldnames:
                if k not in old_fields:
                    old_fields.append(k)
            fieldnames = old_fields
    # 写入CSV文件
    try:
        with open(output_file, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # 补齐缺失字段
            for k in fieldnames:
                if k not in row:
                    row[k] = ""
            writer.writerow(row)
        print(f"算法结果已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")
        return False


def _extract_unified_metrics(stats: Dict, env: Any, config: Any, total_time: float) -> Dict:
    """
    从不同类型的数据中提取统一的性能指标
    自动适配强化学习和启发式算法
    """
    metrics = {
        # 默认值
        'num_jobs': 0,
        'num_machines': 0,
        'num_distributors': 0,
        'num_episodes': 0,
        'total_tardiness': 0.0,
        'weighted_shortage': 0.0,
        'objective_value': 0.0,
        'final_makespan': 0.0,
        'avg_makespan': 0.0,
        'on_time_rate': 0.0,
        'avg_flow_time': 0.0,
        'avg_waiting_time': 0.0,
        'machine_utilization': 0.0,
        'demand_coverage': 0.0,
        'avg_reward': 0.0,
        'std_reward': 0.0,
        'best_reward': 0.0,
        'final_reward': 0.0,
        'time_per_episode_s': 0.0,
        'time_per_step_ms': 0.0,
        'total_steps': 0,
        'algorithm_type': 'Unknown'
    }
    
    # 1. 检测启发式算法结果 (SolutionResult对象)
    if hasattr(env, 'total_objective') and hasattr(env, 'schedule_result'):
        metrics.update(_extract_heuristic_metrics(env, stats, config, total_time))
    
    # 2. 检测强化学习结果
    elif 'episode_rewards' in stats and len(stats['episode_rewards']) > 1:
        metrics.update(_extract_rl_metrics(stats, env, config, total_time))
    
    # 3. 其他情况的通用处理
    else:
        metrics.update(_extract_generic_metrics(stats, env, config, total_time))
    
    return metrics


def _extract_heuristic_metrics(result, stats: Dict, config: Any, total_time: float) -> Dict:
    """提取启发式算法指标"""
    schedule_result = getattr(result, 'schedule_result', None)
    dispatch_result = getattr(result, 'dispatch_result', None)
    
    metrics = {
        'algorithm_type': 'Heuristic',
        'num_episodes': 1,
        'objective_value': float(getattr(result, 'total_objective', 0.0)),
        'time_per_episode_s': total_time,
    }
    
    # 从config获取问题规模
    metrics['num_jobs'] = getattr(config, 'num_jobs', 0)
    metrics['num_machines'] = getattr(config, 'num_machines', 0)
    metrics['num_distributors'] = getattr(config, 'num_distributors', 0)
    
    if schedule_result:
        metrics['total_tardiness'] = float(getattr(schedule_result, 'total_tardiness', 0))
        metrics['final_makespan'] = float(getattr(schedule_result, 'makespan', 0))
        metrics['avg_makespan'] = metrics['final_makespan']
        
        # 机器利用率
        machine_util = getattr(schedule_result, 'machine_utilization', {})
        if machine_util:
            metrics['machine_utilization'] = float(np.mean(list(machine_util.values())))
    
    if dispatch_result:
        metrics['demand_coverage'] = float(getattr(dispatch_result, 'on_time_delivery_rate', 0.0))
    
    return metrics


def _extract_rl_metrics(stats: Dict, env: Any, config: Any, total_time: float) -> Dict:
    """提取强化学习算法指标"""
    episode_rewards = stats.get('episode_rewards', [])
    episode_lengths = stats.get('episode_lengths', [])
    makespans = stats.get('makespans', [])
    
    num_episodes = len(episode_rewards)
    num_steps = int(sum(episode_lengths)) if episode_lengths else 1
    
    metrics = {
        'algorithm_type': 'Reinforcement_Learning',
        'num_episodes': num_episodes,
        'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        'std_reward': float(np.std(episode_rewards)) if episode_rewards else 0.0,
        'best_reward': float(np.max(episode_rewards)) if episode_rewards else 0.0,
        'final_reward': float(episode_rewards[-1]) if episode_rewards else 0.0,
        'final_makespan': float(makespans[-1]) if makespans else 0.0,
        'avg_makespan': float(np.mean(makespans)) if makespans else 0.0,
        'time_per_episode_s': total_time / max(num_episodes, 1),
        'time_per_step_ms': (total_time / num_steps * 1000.0) if num_steps > 0 else 0.0,
        'total_steps': num_steps,
    }
    
    # 从环境获取问题规模
    if hasattr(env, 'jobs'):
        metrics['num_jobs'] = len(env.jobs)
    elif hasattr(env, 'case') and hasattr(env.case, 'jobs'):
        metrics['num_jobs'] = len(env.case.jobs)
    else:
        metrics['num_jobs'] = getattr(config, 'num_jobs', 0)
    
    # 简化其他指标计算
    metrics['num_machines'] = getattr(config, 'num_machines', 0)
    metrics['num_distributors'] = getattr(config, 'num_distributors', 0)
    
    return metrics


def _extract_generic_metrics(stats: Dict, env: Any, config: Any, total_time: float) -> Dict:
    """通用指标提取（兜底方案）"""
    return {
        'algorithm_type': 'Generic',
        'num_jobs': getattr(config, 'num_jobs', 0),
        'num_machines': getattr(config, 'num_machines', 0),
        'num_distributors': getattr(config, 'num_distributors', 0),
        'num_episodes': 1,
        'time_per_episode_s': total_time,
    }


# 保持原有的其他函数不变
def save_convergence_data(
    algo_name: str,
    instance_id: str,
    seed: int,
    stats: Dict,
    config: Any = None,
    additional_series: Optional[Dict[str, List]] = None
):
    """保存收敛数据（保持原有逻辑）"""
    # ... 保持原有实现
    pass


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
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass