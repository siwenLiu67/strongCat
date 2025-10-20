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
def calculate_intrinsic_reward(goal, state, action, env_reward, next_state, info):
    """计算内在奖励：基于子目标完成度"""
    if goal == 0:  # 调度效率目标
        # 基于机器利用率的改进
        current_util = calculate_utilization(state)
        next_util = calculate_utilization(next_state)
        intrinsic = (next_util - current_util) * 0.5
        
    elif goal == 1:  # 配送及时性目标
        # 基于延误减少
        current_tardiness = getattr(state, 'total_tardiness', 0)
        next_tardiness = getattr(next_state, 'total_tardiness', 0)
        intrinsic = max(0, current_tardiness - next_tardiness) * 0.1
        
    else:  # 负载平衡目标
        # 基于负载均衡
        current_balance = calculate_load_balance(state)
        next_balance = calculate_load_balance(next_state)
        intrinsic = (next_balance - current_balance) * 0.3
    
    # 结合环境奖励
    combined_reward = env_reward + intrinsic
    return combined_reward

def calculate_hierarchical_returns(buffer, gamma):
    """计算层次回报"""
    rewards = [data['intrinsic_reward'] for data in buffer]
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def evaluate_goal_completion(goal, state, next_state, info):
    """评估子目标完成度"""
    # 简化的完成度评估
    if goal == 0:  # 调度效率
        return min(1.0, getattr(next_state, 'machine_utilization', 0))
    elif goal == 1:  # 配送及时性
        tardiness = getattr(next_state, 'total_tardiness', 0)
        return max(0, 1 - tardiness / 1000)  # 假设最大延误1000
    else:  # 负载平衡
        return calculate_load_balance(next_state)

def calculate_utilization(state):
    """计算机器利用率"""
    return getattr(state, 'machine_utilization', 0.5)

def calculate_load_balance(state):
    """计算负载均衡度"""
    # 简化的均衡度计算
    return 0.7  # 示例值

class TimeAbstractionLayer:
    """时间抽象层：控制Meta Agent的决策频率"""
    def __init__(self, config):
        self.decision_interval = getattr(config, 'meta_decision_interval', 10)  # 每10步决策一次
        self.steps_since_last_decision = 0
        self.current_strategy = None
        
    def should_decide(self):
        """判断是否需要Meta Agent决策"""
        self.steps_since_last_decision += 1
        if self.steps_since_last_decision >= self.decision_interval:
            self.steps_since_last_decision = 0
            return True
        return False
    
    def set_strategy(self, strategy):
        """设置当前策略"""
        self.current_strategy = strategy
        
    def get_strategy(self):
        """获取当前策略"""
        return self.current_strategy

def run_ruleDqn_dispatchHeuri_experiment(config, case, seed, **kwargs):
    """
    层次强化学习版本 - 最小改动
    """
    set_random_seed(seed)
    config.seed = seed
    instance_id = generate_instance_id(config)
    
    # 初始化环境和智能体
    env = WarehouseEnvironment(config, case)
    meta_agent = MetaAgent(config)
    rule_dqn_agent = RuleBasedDQNAgent(config)
    dispatch_agent = DispatchHeuristic()
    
    # 添加时间抽象层
    time_abstraction = TimeAbstractionLayer(config)
    
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
    hierarchical_buffer = []  # 层次经验缓冲区
    learning_rates = []
    start_time = time.time()
    gamma = getattr(config, "gamma", 0.95)

    for episode in range(config.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_goals = []  # 记录子目标完成情况
        
        # 重置时间抽象层
        time_abstraction = TimeAbstractionLayer(config)
        
        while not done:
            # 层次决策：只在需要时调用Meta Agent
            if time_abstraction.should_decide():
                agent_idx, log_prob, subgoal = meta_agent.select_action(state)
                time_abstraction.set_strategy(agent_idx)
                current_goal = subgoal
            else:
                # 使用缓存的策略
                agent_idx = time_abstraction.get_strategy()
                log_prob = torch.tensor(0.0)  # 占位符
                current_goal = getattr(time_abstraction, 'current_goal', 0)
            
            # 底层执行
            if agent_idx == 0:
                action, rule_idx = rule_dqn_agent.select_action(state)
            elif agent_idx == 1:
                action = dispatch_agent.select_action(state)
                rule_idx = None
            else:
                action = {'wait': True}
                rule_idx = None
                
            # 环境执行
            next_state, reward, done, info = env.step(action)
            
            # RuleDQN在线更新
            if agent_idx == 0:
                rule_dqn_agent.update({
                    'states': [state],
                    'actions': [rule_idx],
                    'rewards': [reward],
                    'next_states': [next_state],
                    'dones': [done]
                })
            
            # 计算内在奖励（层次RL核心）
            intrinsic_reward = calculate_intrinsic_reward(
                current_goal, state, action, reward, next_state, info
            )
            
            # 存储层次经验
            hierarchical_buffer.append({
                'state': state,
                'agent_idx': agent_idx,
                'log_prob': log_prob,
                'env_reward': reward,
                'intrinsic_reward': intrinsic_reward,
                'subgoal': current_goal,
                'step_type': 'meta' if time_abstraction.steps_since_last_decision == 0 else 'execution'
            })
            
            # 记录子目标完成情况
            goal_completion = evaluate_goal_completion(current_goal, state, next_state, info)
            episode_goals.append(goal_completion)
            
            state = next_state
            episode_reward += reward

        # 计算层次回报（结合环境和内在奖励）
        hierarchical_returns = calculate_hierarchical_returns(hierarchical_buffer, gamma)
        
        # Meta Agent批量更新（使用层次回报）
        if len(hierarchical_buffer) >= config.batch_size:
            batch_indices = np.random.choice(
                len(hierarchical_buffer), 
                min(config.batch_size, len(hierarchical_buffer)), 
                replace=False
            )
            
            batch_data = [hierarchical_buffer[i] for i in batch_indices]
            
            # 确保所有数据都是有效的，过滤掉None值
            valid_actions = []
            valid_states = []
            valid_log_probs = []
            valid_returns = []
            
            for i, data in enumerate(batch_data):
                if data['agent_idx'] is not None and data['log_prob'] is not None:
                    valid_actions.append(data['agent_idx'])
                    valid_states.append(data['state'])
                    valid_log_probs.append(data['log_prob'])
                    valid_returns.append(hierarchical_returns[batch_indices[i]])
            
            # 只有在有足够有效数据时才进行更新
            if len(valid_actions) >= 2:  # 至少需要2个样本才能进行有意义的更新
                batch = {
                    'states': valid_states,
                    'actions': valid_actions,
                    'log_probs': valid_log_probs,
                    'returns': torch.tensor(valid_returns, dtype=torch.float32)
                }
                
                meta_loss = meta_agent.update(batch, optimizer, scheduler)
            else:
                meta_loss = 0.0
                print(f"Skipping meta update: only {len(valid_actions)} valid samples available")
            
            # 记录学习率
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            
            # 清理缓冲区（保留部分经验）
            keep_ratio = 0.3  # 保留30%的经验
            keep_size = int(len(hierarchical_buffer) * keep_ratio)
            hierarchical_buffer = hierarchical_buffer[-keep_size:]
        
        # 记录统计信息
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(len(hierarchical_buffer))
        stats['makespans'].append(getattr(env, 't', 0))
        stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
        stats['total_tardiness'].append(getattr(env, 'total_weighted_tardiness', 0))
        stats['machine_utilization'].append(calculate_machine_utilization(env))
        stats['avg_goal_completion'].append(np.mean(episode_goals) if episode_goals else 0)
        
        # 进度输出
        if episode % 10 == 0:
            current_lr = learning_rates[-1] if learning_rates else 0.0003
            avg_goal = stats['avg_goal_completion'][-1]
            print(f'Episode {episode}, Reward: {episode_reward:.2f}, '
                  f'Goal Completion: {avg_goal:.3f}, LR: {current_lr:.6f}')

    # 保存结果
    total_time = time.time() - start_time
    additional_metrics = {
        "algorithm_type": "Hierarchical_RL_RuleDQN_DispatchHeuristic",
        "batch_size": config.batch_size,
        "gamma": gamma,
        "final_learning_rate": learning_rates[-1] if learning_rates else 0.0003,
        "meta_decision_interval": time_abstraction.decision_interval
    }
    
    save_algorithm_results_csv(
        algo_name="Hierarchical_RL_RuleDQN+DispatchHeuristic",
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

# def run_ruleDqn_dispatchHeuri_experiment(config, case, seed, **kwargs):
#     """
#     统一入口，供batch_runner调用
#     """
#     set_random_seed(seed)
#     config.seed = seed
#     instance_id = generate_instance_id(config)
    
#     # 初始化环境和智能体
#     env = WarehouseEnvironment(config, case)
#     meta_agent = MetaAgent(config)
#     rule_dqn_agent = RuleBasedDQNAgent(config)
#     dispatch_agent = DispatchHeuristic()
    
#     # 优化器和调度器
#     optimizer = optim.Adam(meta_agent.parameters(), lr=0.0003)
#     estimated_steps_per_episode = 350
#     total_training_steps = config.episodes * estimated_steps_per_episode
    
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#         optimizer, 
#         max_lr=0.001,
#         total_steps=total_training_steps,
#         pct_start=0.3,
#         div_factor=10.0,
#         final_div_factor=100.0
#     )

#     # 训练状态跟踪
#     stats = defaultdict(list)
#     meta_buffer = defaultdict(list)
#     learning_rates = []
#     start_time = time.time()
#     gamma = getattr(config, "gamma", 0.95)

#     for episode in range(config.episodes):
#         state = env.reset()
#         done = False
#         episode_reward = 0
        
#         while not done:
#             # 层次决策和执行
#             agent_idx, log_prob, _ = meta_agent.select_action(state)
            
#             if agent_idx == 0:
#                 action, rule_idx = rule_dqn_agent.select_action(state)
#             elif agent_idx == 1:
#                 action = dispatch_agent.select_action(state)
#                 rule_idx = None
#             else:
#                 action = {'wait': True}
#                 rule_idx = None
                
#             # 环境执行
#             next_state, reward, done, _ = env.step(action)
            
#             # RuleDQN在线更新
#             if agent_idx == 0:
#                 rule_dqn_agent.update({
#                     'states': [state],
#                     'actions': [rule_idx],
#                     'rewards': [reward],
#                     'next_states': [next_state],
#                     'dones': [done]
#                 })
            
#             # 存储meta经验
#             meta_buffer['states'].append(state)
#             meta_buffer['actions'].append(agent_idx)
#             meta_buffer['rewards'].append(reward)
#             meta_buffer['log_probs'].append(log_prob)
            
#             state = next_state
#             episode_reward += reward

#         # 计算累积回报
#         returns = []
#         R = 0
#         for r in reversed(meta_buffer['rewards']):
#             R = r + gamma * R
#             returns.insert(0, R)
        
#         meta_buffer['returns'].extend(returns)
        
#         # Meta Agent批量更新
#         if len(meta_buffer['states']) >= config.batch_size:
#             batch = {
#                 'states': meta_buffer['states'][:config.batch_size],
#                 'actions': meta_buffer['actions'][:config.batch_size],
#                 'log_probs': meta_buffer['log_probs'][:config.batch_size],
#                 'returns': torch.tensor(meta_buffer['returns'][:config.batch_size], dtype=torch.float32)
#             }
            
#             meta_loss = meta_agent.update(batch, optimizer, scheduler)
            
#             # 记录学习率和清理缓冲区
#             current_lr = scheduler.get_last_lr()[0]
#             learning_rates.append(current_lr)
            
#             # 保留部分经验用于下一轮
#             keep_size = len(meta_buffer['states']) - config.batch_size
#             for key in meta_buffer:
#                 meta_buffer[key] = meta_buffer[key][-keep_size:] if keep_size > 0 else []
        
#         # 记录统计信息
#         stats['episode_rewards'].append(episode_reward)
#         stats['episode_lengths'].append(len(meta_buffer['rewards']))
#         stats['makespans'].append(getattr(env, 't', 0))
#         stats['tardy_penalty'].append(getattr(env, 'tardy_penalty', 0))
#         stats['total_tardiness'].append(getattr(env, 'total_weighted_tardiness', 0))
#         stats['machine_utilization'].append(calculate_machine_utilization(env))
        
#         # 进度输出
#         if episode % 10 == 0:
#             current_lr = learning_rates[-1] if learning_rates else 0.0003
#             print(f'Episode {episode}, Reward: {episode_reward:.2f}, LR: {current_lr:.6f}')

#     # 保存结果
#     total_time = time.time() - start_time
#     additional_metrics = {
#         "algorithm_type": "Hybrid_RuleDQN_DispatchHeuristic",
#         "batch_size": config.batch_size,
#         "gamma": gamma,
#         "final_learning_rate": learning_rates[-1] if learning_rates else 0.0003,
#     }
    
#     save_algorithm_results_csv(
#         algo_name="RuleDQN+DispatchHeuristic",
#         instance_id=instance_id,
#         seed=seed,
#         config=config,
#         stats=stats,
#         env=env,
#         total_time=total_time,
#         additional_metrics=additional_metrics
#     )
    
#     return {
#         'stats': stats,
#         'env': env,
#         'additional_metrics': additional_metrics
#     }

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
