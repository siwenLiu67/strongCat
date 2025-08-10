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

# 导入规范化保存函数
from algorithm_results_saver import save_algorithm_results_csv, generate_instance_id, set_random_seed

def main():
    # 设置随机种子
    seed = 42
    set_random_seed(seed)
    
    # 初始化配置和环境
    config = Config()
    config.seed = seed  # 记录种子
    
    # 生成算例ID
    instance_id = generate_instance_id(config)
    
    # 使用case_generator创建测试场景
    case = FlexibleJobShopScenario(config)
    env = WarehouseEnvironment(config, case)
    
    # 初始化智能体
    meta_agent = MetaAgent(config)
    rule_dqn_agent = RuleBasedDQNAgent(config)
    dispatch_agent = DispatchHeuristic()
    
    # 训练参数
    episodes = 10  # 增加训练轮数
    stats = defaultdict(list)
    batch_size = 32
    
    # 高层智能体经验收集
    meta_states = []
    meta_actions = []
    meta_rewards = []
    meta_log_probs = []
    meta_returns = []
    
    print(f"开始训练 RuleDQN+DispatchHeuristic 算法...")
    print(f"算例: {instance_id}, 种子: {seed}")
    start_time = time.time()
    
    for episode in range(episodes):
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}==========================")
        
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_stats = defaultdict(list)
        
        # 当前episode的数据
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_log_probs = []
        
        while not done:
            # Meta智能体选择当前使用的智能体
            agent_idx, log_prob, _ = meta_agent.select_action(state)
            
            # 根据选择执行对应智能体的决策
            if agent_idx == 0:  # Rule-DQN智能体
                action, rule_idx = rule_dqn_agent.select_action(state)
            elif agent_idx == 1:
                action = dispatch_agent.select_action(state)
                rule_idx = None
            else:
                action = {'wait': True}
                rule_idx = None

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 更新rule-based DQN智能体
            if agent_idx == 0:
                transition = {
                    'states': [state],
                    'actions': [rule_idx],
                    'rewards': [reward],
                    'next_states': [next_state],
                    'dones': [done]
                }
                rule_dqn_agent.update(transition)
            
            # 收集meta智能体的经验
            ep_states.append(state)
            ep_actions.append(agent_idx)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            
            # 更新状态和统计
            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_stats['rewards'].append(reward)
            episode_stats['actions'].append(action)
        
        # 计算累积折扣回报
        returns = []
        R = 0
        gamma = 0.99
        for r in reversed(ep_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # 存入全局经验池
        meta_states.extend(ep_states)
        meta_actions.extend(ep_actions)
        meta_rewards.extend(ep_rewards)
        meta_log_probs.extend(ep_log_probs)
        meta_returns.extend(returns.tolist())
        
        # 批量更新meta智能体
        meta_loss = 0.0
        if len(meta_states) >= batch_size:
            batch = {
                'states': meta_states[:batch_size],
                'actions': meta_actions[:batch_size],
                'log_probs': meta_log_probs[:batch_size],
                'returns': torch.tensor(meta_returns[:batch_size], dtype=torch.float32)
            }
            meta_loss = meta_agent.update(batch)
            
            # 移除已使用的数据
            meta_states = meta_states[batch_size:]
            meta_actions = meta_actions[batch_size:]
            meta_rewards = meta_rewards[batch_size:]
            meta_log_probs = meta_log_probs[batch_size:]
            meta_returns = meta_returns[batch_size:]
        
        # 记录统计数据
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_length)
        stats['makespans'].append(getattr(env, 't', getattr(env, 'current_time', 0)))
        stats['running_times'].append(time.time() - start_time)
        stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
        stats['meta_losses'].append(meta_loss)
        
        # 打印进度
        if (episode + 1) % 20 == 0:
            recent_rewards = stats['episode_rewards'][-20:]
            recent_makespans = stats['makespans'][-20:]
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  平均奖励: {np.mean(recent_rewards):.2f}")
            print(f"  平均makespan: {np.mean(recent_makespans):.2f}")
            print(f"  Meta损失: {meta_loss:.4f}")
    
    total_time = time.time() - start_time
    
    # 算法特定指标
    additional_metrics = {
        "meta_agent_loss": float(np.mean(stats['meta_losses'][-10:])) if stats['meta_losses'] else 0.0,
        "avg_tardy_penalty": float(np.mean(stats['tardy_penalty'])),
        "algorithm_type": "Hybrid_RuleDQN_DispatchHeuristic",
        "batch_size": batch_size,
        "gamma": gamma
    }
    
    # 保存标准化CSV结果
    print(f"\n保存结果到CSV...")
    success = save_algorithm_results_csv(
        algo_name="RuleDQN+DispatchHeuristic",
        instance_id=instance_id,
        seed=seed,
        config=config,
        stats=stats,
        env=env,
        total_time=total_time,
        additional_metrics=additional_metrics
    )
    
    # 保存详细pickle结果（保留原有功能）
    detailed_results = {
        'stats': stats,
        'config': config,
        'instance_id': instance_id,
        'seed': seed,
        'total_time': total_time,
        'additional_metrics': additional_metrics
    }
    
    with open(f'results_detailed_{instance_id}_{seed}.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)
    
    # 输出总结
    print(f"\n{'='*50}")
    print(f"训练完成！")
    print(f"算法: RuleDQN+DispatchHeuristic")
    print(f"算例: {instance_id}")
    print(f"种子: {seed}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均奖励: {np.mean(stats['episode_rewards']):.2f}")
    print(f"平均makespan: {np.mean(stats['makespans']):.2f}")
    print(f"最终奖励: {stats['episode_rewards'][-1]:.2f}")
    print(f"最终makespan: {stats['makespans'][-1]:.2f}")
    
    if success:
        print(f"结果已保存到标准化CSV文件")
    print(f"详细结果已保存到: results_detailed_{instance_id}_{seed}.pkl")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()