from typing import List, Dict, Any, Tuple
import pickle
import torch
from dataclasses import dataclass, field

# Local imports
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
from high_level_agent import HighLevelAgent
from schedule_agent import ScheduleAgent
from dispatch_heuristic import DispatchHeuristic
from improved_schedule_agent import ImprovedScheduleAgent

def run_episode(
    env: WarehouseEnvironment,
    meta_agent: HighLevelAgent,
    scheduling_agent: ImprovedScheduleAgent,
    dispatching_agent: DispatchHeuristic,
    verbose: bool = True
) -> Dict[str, Any]:
    """运行一个完整的训练episode
    
    Args:
        env: 仓库环境实例
        meta_agent: 高层决策智能体
        scheduling_agent: 调度智能体 
        dispatching_agent: 分派启发式算法
        verbose: 是否打印详细信息
        
    Returns:
        包含episode结果的字典，包括奖励、状态、动作等
    """
    state = env.reset()
    done = False
    step = 0
    episode_reward = 0
    rewards = []
    actions = []
    log_probs = []
    states = []
    dispatch_count = 0
    schedule_count = 0

    # 新增：调度相关采集
    schedule_states = []
    schedule_actions = []
    schedule_log_probs = []
    schedule_rewards = []

    while not done:
        # 固定使用调度动作(meta_action=0)
        meta_action = 0
        actions.append(meta_action)
        log_probs.append(torch.tensor(0.0))  # 占位符
        states.append(state)
        
        # 调度动作采集
        schedule_action, schedule_log_prob = scheduling_agent.select_action(state, return_log_prob=True)
        action = {'schedule': schedule_action} if schedule_action else {'wait': True}
        schedule_count += 1
        # 记录调度数据
        schedule_states.append(state)
        schedule_actions.append(schedule_action)
        schedule_log_probs.append(schedule_log_prob)

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        # 新增：调度奖励采集（可用主reward或自定义）
        if meta_action == 0:
            schedule_rewards.append(reward)
        state = next_state
        step += 1

    # 计算 schedule_returns（折扣累计奖励）
    schedule_returns = []
    G = 0
    gamma = 0.99
    for r in reversed(schedule_rewards):
        G = r + gamma * G
        schedule_returns.insert(0, G)
    schedule_returns = torch.tensor(schedule_returns, dtype=torch.float32)

    # 其他统计...
    completed_jobs = getattr(env, 'completed_jobs', []) or state.get('completed_jobs', [])
    finish_times = [getattr(job, 'completed_time', 0) for job in completed_jobs]
    due_times = [getattr(job, 'due_time', 0) for job in completed_jobs]
    lateness = [max(0, ft - dt) for ft, dt in zip(finish_times, due_times)]
    makespan = max(finish_times) if finish_times else 0
    total_late_jobs = sum(1 for l in lateness if l > 0)
    total_late_time = sum(lateness)

    return {
        "episode_reward": episode_reward,
        "rewards": rewards,
        "actions": actions,
        "log_probs": log_probs,
        "returns": torch.tensor(rewards, dtype=torch.float32),
        "states": states,
        "dispatch_count": dispatch_count,
        "schedule_count": schedule_count,
        "steps": step,
        "makespan": makespan,
        "total_late_jobs": total_late_jobs,
        "total_late_time": total_late_time,
        # 新增调度相关
        "schedule_states": schedule_states,
        "schedule_actions": schedule_actions,
        "schedule_log_probs": schedule_log_probs,
        "schedule_rewards": schedule_rewards,
        "schedule_returns": schedule_returns,
    }



def collect_episode_stats() -> Tuple[
    List[float], List[int], List[int], List[int], 
    List[List[float]], List[float], List[int], List[float]
]:
    """收集并初始化所有episode统计数据的容器
    
    Returns:
        包含所有统计容器的元组:
        - all_rewards: 每episode总奖励
        - all_steps: 每episode步数
        - all_dispatch_counts: 每episode分派次数
        - all_schedule_counts: 每episode调度次数
        - all_rewards_per_episode: 每episode每步奖励
        - all_makespans: 每episode最大完成时间
        - all_total_late_jobs: 每episode延迟任务数
        - all_total_late_time: 每episode总延迟时间
    """
    return [], [], [], [], [], [], [], []

def print_episode_stats(
    all_rewards: List[float],
    all_steps: List[int],
    all_dispatch_counts: List[int],
    all_schedule_counts: List[int]
) -> None:
    """打印episode统计摘要
    
    Args:
        all_rewards: 每episode总奖励列表
        all_steps: 每episode步数列表
        all_dispatch_counts: 每episode分派次数列表
        all_schedule_counts: 每episode调度次数列表
    """
    print("\n========== 统计 ==========")
    print(f"所有episode奖励: {all_rewards}")
    print(f"平均奖励: {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"平均步数: {sum(all_steps)/len(all_steps):.2f}")
    print(f"平均dispatch次数: {sum(all_dispatch_counts)/len(all_dispatch_counts):.2f}")
    print(f"平均schedule次数: {sum(all_schedule_counts)/len(all_schedule_counts):.2f}")

def save_results(results: Dict[str, Any], filename: str = 'results.pkl') -> None:
    """保存训练结果到文件
    
    Args:
        results: 要保存的结果字典
        filename: 保存文件名
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def train_agents(
    scheduling_agent: ImprovedScheduleAgent,
    ep_result: Dict[str, Any],
    ep: int
) -> None:
    """训练调度智能体
    
    Args:
        scheduling_agent: 调度智能体
        ep_result: 包含训练数据的episode结果
    """
    # 训练scheduling_agent
    schedule_batch = {
        'states': ep_result['schedule_states'],
        'actions': ep_result['schedule_actions'],
        'log_probs': ep_result['schedule_log_probs'],
        'returns': ep_result['schedule_returns'],
    }
    scheduling_agent.update()

def main():
    """主训练流程"""
    config = Config()
    num_episodes = 20
    
    # 初始化统计容器
    (all_rewards, all_steps, all_dispatch_counts, 
     all_schedule_counts, all_rewards_per_episode,
     all_makespans, all_total_late_jobs, 
     all_total_late_time) = collect_episode_stats()

    for ep in range(num_episodes):
        print(f"\n================ Episode {ep+1} ================")
        case = FlexibleJobShopScenario(config)
        env = WarehouseEnvironment(config, case)
        meta_agent = HighLevelAgent(config)
        scheduling_agent = ImprovedScheduleAgent(config)
        dispatching_agent = DispatchHeuristic()

        ep_result = run_episode(
            env, meta_agent, scheduling_agent, dispatching_agent, verbose=False
        )


        # 训练智能体
        train_agents(scheduling_agent, ep_result, ep)

        print(f"Episode {ep+1} 总奖励: {ep_result['episode_reward']:.2f}")
        all_rewards.append(ep_result['episode_reward'])
        all_steps.append(ep_result['steps'])
        all_dispatch_counts.append(ep_result['dispatch_count'])
        all_schedule_counts.append(ep_result['schedule_count'])
        all_rewards_per_episode.append(ep_result['rewards'])
        all_makespans.append(ep_result['makespan'])
        all_total_late_jobs.append(ep_result['total_late_jobs'])
        all_total_late_time.append(ep_result['total_late_time'])

    print_episode_stats(all_rewards, all_steps, all_dispatch_counts, all_schedule_counts)

    # 保存结果
    save_results({
        'all_rewards': all_rewards,
        'all_steps': all_steps,
        'all_dispatch_counts': all_dispatch_counts,
        'all_schedule_counts': all_schedule_counts,
        'all_rewards_per_episode': all_rewards_per_episode,
        'all_makespans': all_makespans,
        'all_total_late_jobs': all_total_late_jobs,
        'all_total_late_time': all_total_late_time,
    })

if __name__ == "__main__":
    main()
