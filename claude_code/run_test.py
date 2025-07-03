from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
from high_level_agent import HighLevelAgent
from schedule_agent import ScheduleAgent
from dispatch_agent import DispatchAgent


def run_episode(env, meta_agent, scheduling_agent, dispatching_agent, verbose=True):
    """运行单个episode，返回累计奖励"""
    state = env.reset()
    done = False
    step = 0
    episode_reward = 0

    while not done:
        meta_action, _ = meta_agent.select_action(state)
        if meta_action == 0:
            action = {'schedule': scheduling_agent.select_action(state)}
            if not action['schedule']:
                action = {'wait': True}
        elif meta_action == 1:
            completed = [job.job_id for job in state['completed_jobs'] if hasattr(job, 'status') and job.status == 'completed']
            if completed:
                action = {'dispatch': dispatching_agent.select_action(state, completed_jobs=completed)}
            else:
                action = {'wait': True}
        else:
            action = {'wait': True}

        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        if verbose:
            print(f"\n时间步 {step}:")
            print(f"高层动作: {meta_action}，执行动作: {action}")
            print(f"- 可用作业数量: {len(next_state['available_jobs'])}")
            print(f"- 完成作业数量: {len(next_state['completed_jobs'])}")
            print(f"- 配送作业数量: {len(next_state.get('dispatched_jobs', []))}")
            print(f"- 奖励值: {reward:.2f}")

        state = next_state
        step += 1

    if verbose:
        print("\n环境结束!")
    return episode_reward

def main():
    config = Config()
    case = FlexibleJobShopScenario(config)
    print("生成的FJSP-DP算例:")
    print(f"作业数量: {len(case.jobs)}")
    print(f"机器数量: {len(case.machines)}")
    print(f"配送商数量: {len(case.distributors)}")

    meta_agent = HighLevelAgent(config)
    scheduling_agent = ScheduleAgent(config)
    dispatching_agent = DispatchAgent(config)

    num_episodes =2
    all_rewards = []

    for ep in range(num_episodes):
        print(f"\n========== Episode {ep+1} ==========")

        case = FlexibleJobShopScenario(config)  # 每次新建
        env = WarehouseEnvironment(config, case)
        ep_reward = run_episode(env, meta_agent, scheduling_agent, dispatching_agent,
                                 verbose=True)
        print(f"Episode {ep+1} 总奖励: {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    print("\n========== 统计 ==========")
    print(f"所有episode奖励: {all_rewards}")
    print(f"平均奖励: {sum(all_rewards)/len(all_rewards):.2f}")

if __name__ == "__main__":
    main()