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


def run_episode(
    env,
    meta_agent,
    scheduling_agent,
    dispatching_agent,
    verbose: bool = True
) -> float:
    """运行单个episode，返回累计奖励"""
    state = env.reset()
    done = False
    step = 0
    episode_reward = 0

    while not done:
        meta_action, _ = meta_agent.select_action(state)
        if meta_action == 0:
            schedule = scheduling_agent.select_action(state)
            action = {'schedule': schedule} if schedule else {'wait': True}
        elif meta_action == 1:
            # 兼容 DispatchHeuristic 只需 state 参数
            action = {'dispatch': dispatching_agent.select_action(state)}
        else:
            action = {'wait': True}

        next_state, reward, done, info = env.step(action)
        episode_reward += reward
            
        state = next_state
        step += 1

    if verbose:
        print("\n环境结束!")
    return episode_reward

def main():
    config = Config()
    num_episodes = 1
    all_rewards = []

    for ep in range(num_episodes):
        print(f"\n================ Episode {ep+1} ================")
        case = FlexibleJobShopScenario(config)
        env = WarehouseEnvironment(config, case)
        meta_agent = HighLevelAgent(config)
        scheduling_agent = ScheduleAgent(config)
        dispatching_agent = DispatchAgent(config)

        ep_reward = run_episode(
            env, meta_agent, scheduling_agent, dispatching_agent, verbose=True
        )
        print(f"Episode {ep+1} 总奖励: {ep_reward:.2f}")
        all_rewards.append(ep_reward)

    print("\n========== 统计 ==========")
    print(f"所有episode奖励: {all_rewards}")
    print(f"平均奖励: {sum(all_rewards)/len(all_rewards):.2f}")

if __name__ == "__main__":
    main()