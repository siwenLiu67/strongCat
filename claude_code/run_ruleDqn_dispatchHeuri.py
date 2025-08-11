import sys
import os
import time
import pickle
import numpy as np
from collections import defaultdict
import torch

from environment import WarehouseEnvironment
from rule_based_agent import RuleBasedDQNAgent 
from dispatch_heuristic import DispatchHeuristic
from high_level_agent import HighLevelAgent as MetaAgent
from data_structures import Job
from case_generator import FlexibleJobShopScenario
from config import Config

from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed

def run_ruleDqn_dispatchHeuri_experiment(config, case, seed, **kwargs):
    """
    统一入口，供batch_runner调用
    """
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    env = WarehouseEnvironment(config, case)
    meta_agent = MetaAgent(config)
    rule_dqn_agent = RuleBasedDQNAgent(config)
    dispatch_agent = DispatchHeuristic()
    episodes = getattr(config, "episodes", 1)
    stats = defaultdict(list)
    batch_size = getattr(config, "batch_size", 32)
    meta_states, meta_actions, meta_rewards, meta_log_probs, meta_returns = [], [], [], [], []
    start_time = time.time()
    gamma = getattr(config, "gamma", 0.95)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        ep_states, ep_actions, ep_rewards, ep_log_probs = [], [], [], []
        while not done:
            agent_idx, log_prob, _ = meta_agent.select_action(state)
            if agent_idx == 0:
                action, rule_idx = rule_dqn_agent.select_action(state)
            elif agent_idx == 1:
                action = dispatch_agent.select_action(state)
                rule_idx = None
            else:
                action = {'wait': True}
                rule_idx = None
            next_state, reward, done, info = env.step(action)
            if agent_idx == 0:
                transition = {
                    'states': [state],
                    'actions': [rule_idx],
                    'rewards': [reward],
                    'next_states': [next_state],
                    'dones': [done]
                }
                rule_dqn_agent.update(transition)
            ep_states.append(state)
            ep_actions.append(agent_idx)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            state = next_state
            episode_reward += reward
            episode_length += 1
        # 累积回报
        returns = []
        R = 0
        for r in reversed(ep_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        meta_states.extend(ep_states)
        meta_actions.extend(ep_actions)
        meta_rewards.extend(ep_rewards)
        meta_log_probs.extend(ep_log_probs)
        meta_returns.extend(returns.tolist())
        meta_loss = 0.0
        if len(meta_states) >= batch_size:
            batch = {
                'states': meta_states[:batch_size],
                'actions': meta_actions[:batch_size],
                'log_probs': meta_log_probs[:batch_size],
                'returns': torch.tensor(meta_returns[:batch_size], dtype=torch.float32)
            }
            meta_loss = meta_agent.update(batch)
            meta_states = meta_states[batch_size:]
            meta_actions = meta_actions[batch_size:]
            meta_rewards = meta_rewards[batch_size:]
            meta_log_probs = meta_log_probs[batch_size:]
            meta_returns = meta_returns[batch_size:]
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(getattr(env, 't', getattr(env, 'current_time', 0)))
        stats['running_times'].append(time.time() - start_time)
        stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
        stats['meta_losses'].append(meta_loss)
    total_time = time.time() - start_time
    additional_metrics = {
        "meta_agent_loss": float(np.mean(stats['meta_losses'][-10:])) if stats['meta_losses'] else 0.0,
        "avg_tardy_penalty": float(np.mean(stats['tardy_penalty'])),
        "algorithm_type": "Hybrid_RuleDQN_DispatchHeuristic",
        "batch_size": batch_size,
        "gamma": gamma
    }
    save_algorithm_results_csv(
        algo_name="RuleDQN+DispatchHeuristic",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time,
        additional_metrics=additional_metrics
    )
    result = {
        'stats': stats,
        'env': env,
        'additional_metrics': additional_metrics
    }
    return result

# 保留原main函数用于单独运行
def main():
    seed = 42
    set_random_seed(seed)
    config = Config()
    case = FlexibleJobShopScenario(config)
    result = run_ruleDqn_dispatchHeuri_experiment(config, case, seed)
    print("实验完成，结果已保存。")

if __name__ == "__main__":
    main()