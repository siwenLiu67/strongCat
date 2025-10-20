import time
import numpy as np
from collections import defaultdict
import torch
import torch.optim as optim

from environment import WarehouseEnvironment
from rule_based_agent import RuleBasedDQNAgent 
from dispatch_heuristic import DispatchHeuristic
from high_level_agent import HighLevelAgent as MetaAgent
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
    
    # 初始化环境和智能体
    env = WarehouseEnvironment(config, case)
    meta_agent = MetaAgent(config)
    rule_dqn_agent = RuleBasedDQNAgent(config)
    dispatch_agent = DispatchHeuristic()
    
    # 优化器和调度器
    optimizer = optim.Adam(meta_agent.parameters(), lr=0.0003)
    estimated_steps_per_episode = 350
    total_training_steps = config.episodes * estimated_steps_per_episode
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001,
        total_steps=total_training_steps,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0
    )

    # 训练状态跟踪
    stats = defaultdict(list)
    meta_buffer = defaultdict(list)
    learning_rates = []
    start_time = time.time()
    gamma = getattr(config, "gamma", 0.95)

    for episode in range(config.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 层次决策和执行
            agent_idx, log_prob, _ = meta_agent.select_action(state)
            
            if agent_idx == 0:
                action, rule_idx = rule_dqn_agent.select_action(state)
            elif agent_idx == 1:
                action = dispatch_agent.select_action(state)
                rule_idx = None
            else:
                action = {'wait': True}
                rule_idx = None
                
            # 环境执行
            next_state, reward, done, _ = env.step(action)
            
            # RuleDQN在线更新
            if agent_idx == 0:
                rule_dqn_agent.update({
                    'states': [state],
                    'actions': [rule_idx],
                    'rewards': [reward],
                    'next_states': [next_state],
                    'dones': [done]
                })
            
            # 存储meta经验
            meta_buffer['states'].append(state)
            meta_buffer['actions'].append(agent_idx)
            meta_buffer['rewards'].append(reward)
            meta_buffer['log_probs'].append(log_prob)
            
            state = next_state
            episode_reward += reward

        # 计算累积回报
        returns = []
        R = 0
        for r in reversed(meta_buffer['rewards']):
            R = r + gamma * R
            returns.insert(0, R)
        
        meta_buffer['returns'].extend(returns)
        
        # Meta Agent批量更新
        if len(meta_buffer['states']) >= config.batch_size:
            batch = {
                'states': meta_buffer['states'][:config.batch_size],
                'actions': meta_buffer['actions'][:config.batch_size],
                'log_probs': meta_buffer['log_probs'][:config.batch_size],
                'returns': torch.tensor(meta_buffer['returns'][:config.batch_size], dtype=torch.float32)
            }
            
            meta_loss = meta_agent.update(batch, optimizer, scheduler)
            
            # 记录学习率和清理缓冲区
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            
            # 保留部分经验用于下一轮
            keep_size = len(meta_buffer['states']) - config.batch_size
            for key in meta_buffer:
                meta_buffer[key] = meta_buffer[key][-keep_size:] if keep_size > 0 else []
        
        # 记录统计信息
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(len(meta_buffer['rewards']))
        stats['makespans'].append(getattr(env, 't', 0))
        stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
        stats['total_tardiness'].append(getattr(env, 'total_weighted_tardiness', 0))
        stats['machine_utilization'].append(calculate_machine_utilization(env))
        
        # 进度输出
        if episode % 10 == 0:
            current_lr = learning_rates[-1] if learning_rates else 0.0003
            print(f'Episode {episode}, Reward: {episode_reward:.2f}, LR: {current_lr:.6f}')

    # 保存结果
    total_time = time.time() - start_time
    additional_metrics = {
        "algorithm_type": "Hybrid_RuleDQN_DispatchHeuristic",
        "batch_size": config.batch_size,
        "gamma": gamma,
        "final_learning_rate": learning_rates[-1] if learning_rates else 0.0003,
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
    
    return {
        'stats': stats,
        'env': env,
        'additional_metrics': additional_metrics
    }

def calculate_machine_utilization(env) -> float:
    """计算机器利用率"""
    makespan = env.t if env.t > 0 else 1
    total_work_time = sum(getattr(m, 'total_busy_time', 0) for m in env.machines)
    total_available_time = makespan * len(env.machines)
    return total_work_time / total_available_time if total_available_time > 0 else 0.0

def main():
    """主函数用于单独运行"""
    seed = 42
    set_random_seed(seed)
    config = Config()
    case = FlexibleJobShopScenario(config=config)
    result = run_ruleDqn_dispatchHeuri_experiment(config, case, seed)
    print("实验完成，结果已保存。")

if __name__ == "__main__":
    main()