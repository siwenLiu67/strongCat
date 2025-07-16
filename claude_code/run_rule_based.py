from typing import List, Dict, Any, Tuple
import pickle
import torch
import random
import numpy as np

# Local imports
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
from high_level_agent import HighLevelAgent
from rule_based_agent import RuleBasedDQNAgent
from dispatch_heuristic import DispatchHeuristic

def run_episode(
    env: WarehouseEnvironment,
    meta_agent: HighLevelAgent,
    scheduling_agent: RuleBasedDQNAgent,
    dispatching_agent: DispatchHeuristic,
    verbose: bool = True
) -> Dict[str, Any]:
    """运行一个完整的训练episode
    
    Args:
        env: 仓库环境实例
        meta_agent: 高层决策智能体
        scheduling_agent: 基于规则的DQN调度智能体
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
    states = []
    next_states = []
    actions = []
    dones = []
    dispatch_count = 0
    schedule_count = 0

    while not done:
        # 固定使用调度动作(meta_action=0)
        meta_action = 0
        actions.append(meta_action)
        states.append(state)
        
        # DQN选择调度规则
        schedule_action = scheduling_agent.select_action(state)
        action = schedule_action if schedule_action else {'wait': True}
        schedule_count += 1

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        # 存储DQN经验
        # 处理schedule_action可能为空或没有'schedule'键的情况
        action_value = 0
        if schedule_action and 'schedule' in schedule_action and schedule_action['schedule']:
            action_value = list(schedule_action['schedule'].values())[0]
            
        transition = {
            'state': scheduling_agent._get_state(state),
            'action': action_value,
            'reward': reward,
            'next_state': scheduling_agent._get_state(next_state),
            'done': done
        }
        scheduling_agent.replay_buffer.push(**transition)
        
        state = next_state
        step += 1

    # 统计信息
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
        "states": states,
        "next_states": next_states,
        "dones": dones,
        "dispatch_count": dispatch_count,
        "schedule_count": schedule_count,
        "steps": step,
        "makespan": makespan,
        "total_late_jobs": total_late_jobs,
        "total_late_time": total_late_time,
    }

def collect_episode_stats() -> Tuple[
    List[float], List[float], List[int], List[int], List[int], 
    List[List[float]], List[float], List[int], List[float]
]:
    """收集并初始化所有episode统计数据的容器
    
    Returns:
        包含所有统计容器的元组:
        - all_rewards: 每episode总奖励
        - all_losses: 每episode损失
        - all_steps: 每episode步数
        - all_dispatch_counts: 每episode分派次数
        - all_schedule_counts: 每episode调度次数
        - all_rewards_per_episode: 每episode每步奖励
        - all_makespans: 每episode最大完成时间
        - all_total_late_jobs: 每episode延迟任务数
        - all_total_late_time: 每episode总延迟时间
    """
    return [], [], [], [], [], [], [], [], []

def print_episode_stats(
    all_rewards: List[float],
    all_losses: List[float],
    all_steps: List[int],
    all_dispatch_counts: List[int],
    all_schedule_counts: List[int]
) -> None:
    """打印episode统计摘要
    
    Args:
        all_rewards: 每episode总奖励列表
        all_losses: 每episode损失列表
        all_steps: 每episode步数列表
        all_dispatch_counts: 每episode分派次数列表
        all_schedule_counts: 每episode调度次数列表
    """
    print("\n========== 统计 ==========")
    print(f"所有episode奖励: {all_rewards}")
    print(f"所有episode的loss值: {all_losses}")
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

def train_dqn_agent(
    agent: RuleBasedDQNAgent,
    batch_size: int = 32
) -> float:
    """训练DQN智能体
    
    Args:
        agent: DQN智能体实例
        batch_size: 训练batch大小
    """
    if len(agent.replay_buffer) >= batch_size:
        states, actions, rewards, next_states, dones = agent.replay_buffer.sample(batch_size)
        transition_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        return agent.update(transition_dict)
    else:
        return 0.0  # 如果经验不足，返回0损失

def main():
    """主训练流程"""
    config = Config()
    num_episodes = 100
    
    # 初始化统计容器
    (all_rewards, all_losses, all_steps, all_dispatch_counts, 
     all_schedule_counts, all_rewards_per_episode,
     all_makespans, all_total_late_jobs, 
     all_total_late_time) = collect_episode_stats()
    
    meta_agent = HighLevelAgent(config)
    scheduling_agent = RuleBasedDQNAgent(config)
    dispatching_agent = DispatchHeuristic()

    for ep in range(num_episodes):
        print(f"\n================ Episode {ep+1} ================")
        case = FlexibleJobShopScenario(config)
        env = WarehouseEnvironment(config, case)
        
        ep_result = run_episode(
            env, meta_agent, scheduling_agent, dispatching_agent, verbose=False
        )

        # 训练DQN智能体
        loss = train_dqn_agent(scheduling_agent)

        print(f"Episode {ep+1} 总奖励: {ep_result['episode_reward']:.2f}")
        all_rewards.append(ep_result['episode_reward'])
        all_losses.append(loss)
        all_steps.append(ep_result['steps'])
        all_dispatch_counts.append(ep_result['dispatch_count'])
        all_schedule_counts.append(ep_result['schedule_count'])
        all_rewards_per_episode.append(ep_result['rewards'])
        all_makespans.append(ep_result['makespan'])
        all_total_late_jobs.append(ep_result['total_late_jobs'])
        all_total_late_time.append(ep_result['total_late_time'])

    print_episode_stats(all_rewards,all_losses, all_steps, all_dispatch_counts, all_schedule_counts)

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
