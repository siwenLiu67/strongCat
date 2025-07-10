from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
from high_level_agent import HighLevelAgent
from schedule_agent import ScheduleAgent
from dispatch_agent import DispatchAgent
from dispatch_heuristic import DispatchHeuristic
import pickle

def run_episode(
    env,
    meta_agent,
    scheduling_agent,
    dispatching_agent,
    verbose: bool = True
) -> Dict[str, Any]:
    """运行单个episode，返回指标数据"""
    state = env.reset()
    done = False
    step = 0
    episode_reward = 0
    rewards = []
    actions = []
    dispatch_count = 0
    schedule_count = 0

    while not done:
        meta_action, _ = meta_agent.select_action(state)
        actions.append(meta_action)
        if meta_action == 0:
            schedule = scheduling_agent.select_action(state)
            action = {'schedule': schedule} if schedule else {'wait': True}
            schedule_count += 1
        elif meta_action == 1:
            action = {'dispatch': dispatching_agent.select_action(state)}
            dispatch_count += 1
        else:
            action = {'wait': True}

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        rewards.append(reward)
        state = next_state
        step += 1

    if verbose:
        print("\n环境结束!")

    # episode结束后统计
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
        "dispatch_count": dispatch_count,
        "schedule_count": schedule_count,
        "steps": step,
        "makespan": makespan,
        "total_late_jobs": total_late_jobs,
        "total_late_time": total_late_time,
    }

def main():
    config = Config()
    num_episodes = 1
    all_rewards = []
    all_steps = []
    all_dispatch_counts = []
    all_schedule_counts = []
    all_rewards_per_episode = []
    all_makespans = []
    all_total_late_jobs = []
    all_total_late_time = []


    for ep in range(num_episodes):
        print(f"\n================ Episode {ep+1} ================")
        case = FlexibleJobShopScenario(config)
        env = WarehouseEnvironment(config, case)
        meta_agent = HighLevelAgent(config)
        scheduling_agent = ScheduleAgent(config)
        dispatching_agent = DispatchHeuristic()

        ep_result = run_episode(
            env, meta_agent, scheduling_agent, dispatching_agent, verbose=False
        )
        print(f"Episode {ep+1} 总奖励: {ep_result['episode_reward']:.2f}")
        all_rewards.append(ep_result['episode_reward'])
        all_steps.append(ep_result['steps'])
        all_dispatch_counts.append(ep_result['dispatch_count'])
        all_schedule_counts.append(ep_result['schedule_count'])
        all_rewards_per_episode.append(ep_result['rewards'])
        all_makespans.append(ep_result['makespan'])
        all_total_late_jobs.append(ep_result['total_late_jobs'])
        all_total_late_time.append(ep_result['total_late_time'])

    print("\n========== 统计 ==========")
    print(f"所有episode奖励: {all_rewards}")
    print(f"平均奖励: {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"平均步数: {sum(all_steps)/len(all_steps):.2f}")
    print(f"平均dispatch次数: {sum(all_dispatch_counts)/len(all_dispatch_counts):.2f}")
    print(f"平均schedule次数: {sum(all_schedule_counts)/len(all_schedule_counts):.2f}")

    # 保存
    results = {
        'all_rewards': all_rewards,
        'all_steps': all_steps,
        'all_dispatch_counts': all_dispatch_counts,
        'all_schedule_counts': all_schedule_counts,
        'all_rewards_per_episode': all_rewards_per_episode,
        'all_makespans': all_makespans,
        'all_total_late_jobs': all_total_late_jobs,
        'all_total_late_time': all_total_late_time,
    }
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()