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

def main():
    # 初始化配置和环境
    config = Config()
    # 使用case_generator创建测试场景
    case = FlexibleJobShopScenario(config)
    env = WarehouseEnvironment(config, case)
    
    # 初始化智能体
    meta_agent = MetaAgent(config)
    rule_dqn_agent = RuleBasedDQNAgent(config)
    dispatch_agent = DispatchHeuristic()
    
    # 训练参数
    episodes = 2  # 默认训练10个episode
    stats = defaultdict(list)
    batch_size = 32  # 批量更新大小
    
    # 高层智能体经验收集
    meta_states = []
    meta_actions = []
    meta_rewards = []
    meta_log_probs = []
    meta_returns = []
    
    print("开始训练...")
    start_time = time.time()
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}==========================")
        state = env.reset()
        done = False
        episode_reward = 0
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
                    'actions': [rule_idx],  # 使用规则索引，而不是动作字典
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
            
            # 记录数据
            state = next_state
            episode_reward += reward
            episode_stats['rewards'].append(reward)
            episode_stats['actions'].append(action)
        
        # 计算每步的累积折扣回报
        returns = []
        R = 0
        gamma = 0.99  # 折扣因子
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
        if len(meta_states) >= batch_size:
            batch = {
                'states': meta_states[:batch_size],
                'actions': meta_actions[:batch_size],
                'log_probs': meta_log_probs[:batch_size],
                'returns': torch.tensor(meta_returns[:batch_size], dtype=torch.float32)  # 转换为张量
            }
            meta_loss = meta_agent.update(batch)
            # 移除已使用的数据
            meta_states = meta_states[batch_size:]
            meta_actions = meta_actions[batch_size:]
            meta_rewards = meta_rewards[batch_size:]
            meta_log_probs = meta_log_probs[batch_size:]
            meta_returns = meta_returns[batch_size:]
            print(f"Meta Agent Loss: {meta_loss:.4f}")
        
        # 记录每轮数据
        stats['episode_rewards'].append(episode_reward)
        stats['makespans'].append(env.t)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward:.2f}")
    
    # 保存结果
    with open('results.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"训练完成，耗时: {time.time() - start_time:.2f}秒")


if __name__ == '__main__':
    main()
