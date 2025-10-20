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

def analyze_strategic_context(state):
    """分析当前状态，为高层决策提供战略依据"""
    strategic_info = {}
    
    # 正确获取关键指标 - 使用字典访问
    machine_utilization = state.get('machine_utilization', 0.5)
    completed_jobs = len(state.get('completed_jobs', []))
    active_jobs = len(state.get('available_jobs', []))
    total_tardiness = state.get('total_tardiness', 0)
    
    # 分析战略需求
    strategic_info['needs_production'] = (
        machine_utilization < 0.7 or  # 产能利用率低
        active_jobs > completed_jobs   # 积压作业多
    )
    
    strategic_info['needs_delivery'] = (
        completed_jobs >= 5 or         # 完成作业积累较多
        total_tardiness > 100          # 延误风险高
    )
    
    strategic_info['should_wait'] = (
        machine_utilization > 0.9 and  # 机器繁忙
        completed_jobs < 3             # 完成作业少
    )
    
    return strategic_info

def infer_strategic_goal(agent_idx, state):
    """根据选择的动作和当前状态推断战略目标"""
    strategic_info = analyze_strategic_context(state)
    
    if agent_idx == 0:  # 生产模式
        if strategic_info['needs_production']:
            machine_utilization = state.get('machine_utilization', 0.5)
            if machine_utilization < 0.7:
                return "提高产能利用率"
            else:
                return "处理作业积压"
        else:
            return "正常生产调度"
            
    elif agent_idx == 1:  # 配送模式
        if strategic_info['needs_delivery']:
            completed_count = len(state.get('completed_jobs', []))
            total_tardiness = state.get('total_tardiness', 0)
            if completed_count >= 8:
                return "大批次经济派送"
            elif total_tardiness > 100:
                return "紧急防延误派送"
            else:
                return "常规积压清理"
        else:
            return "预防性派送"
            
    else:  # 等待模式
        if strategic_info['should_wait']:
            return "避免过度生产"
        else:
            return "系统平衡等待"

def calculate_goal_based_reward(strategic_goal, prev_state, next_state, env_reward):
    """基于战略目标完成度的奖励"""
    intrinsic = 0.0
    
    if strategic_goal == "提高产能利用率":
        prev_util = prev_state.get('machine_utilization', 0.5)
        next_util = next_state.get('machine_utilization', 0.5)
        intrinsic = max(0, next_util - prev_util) * 20.0
            
    elif strategic_goal == "处理作业积压":
        prev_active = len(prev_state.get('active_jobs', []))
        next_active = len(next_state.get('active_jobs', []))
        intrinsic = max(0, prev_active - next_active) * 5.0
            
    elif strategic_goal == "大批次经济派送":
        prev_completed = len(prev_state.get('completed_jobs', []))
        next_completed = len(next_state.get('completed_jobs', []))
        if prev_completed - next_completed >= 5:  # 大批次派送
            intrinsic = 8.0
        else:
            intrinsic = 2.0
                
    elif strategic_goal == "紧急防延误派送":
        prev_tardiness = prev_state.get('total_tardiness', 0)
        next_tardiness = next_state.get('total_tardiness', 0)
        intrinsic = max(0, prev_tardiness - next_tardiness) * 1.0
            
    elif strategic_goal == "避免过度生产":
        # 等待模式的奖励：避免在高压时增加负担
        next_util = next_state.get('machine_utilization', 0.5)
        if next_util > 0.9:
            intrinsic = 3.0  # 正确等待
        else:
            intrinsic = -1.0  # 不必要等待
                
    else:  # 默认目标
        intrinsic = env_reward * 0.1
    
    # 战略奖励权重较高，因为决策影响更大
    return env_reward + intrinsic

def calculate_hierarchical_returns(buffer, gamma):
    """计算层次回报"""
    rewards = [data['intrinsic_reward'] for data in buffer]
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def evaluate_strategic_success(strategic_goal, state, next_state):
    """评估战略目标完成度"""
    if strategic_goal == "提高产能利用率":
        next_util = next_state.get('machine_utilization', 0.5)
        return min(1.0, next_util)
            
    elif strategic_goal == "处理作业积压":
        active_count = len(next_state.get('active_jobs', []))
        return max(0, 1 - active_count / 20)  # 假设最大20个活跃作业
            
    elif "派送" in strategic_goal:
        completed_count = len(next_state.get('completed_jobs', []))
        total_tardiness = next_state.get('total_tardiness', 0)
        completion_score = min(1.0, completed_count / 15)  # 完成度
        timeliness_score = max(0, 1 - total_tardiness / 500)  # 及时性
        return (completion_score + timeliness_score) / 2
            
    else:  # 等待模式
        next_util = next_state.get('machine_utilization', 0.5)
        return 0.8 if next_util > 0.85 else 0.3  # 高利用率时等待是成功的

class TimeAbstractionLayer:
    """时间抽象层：控制Meta Agent的决策频率"""
    def __init__(self, config):
        self.decision_interval = config.meta_decision_interval
        self.steps_since_last_decision = 0
        self.current_strategy = None
        self.current_strategic_goal = "正常生产调度"  # 默认目标
        
    def should_decide(self):
        """判断是否需要Meta Agent决策"""
        self.steps_since_last_decision += 1
        if self.steps_since_last_decision >= self.decision_interval:
            self.steps_since_last_decision = 0
            return True
        return False
    
    def set_strategy_and_goal(self, strategy, goal):
        """设置当前策略和战略目标"""
        self.current_strategy = strategy
        self.current_strategic_goal = goal
        
    def get_strategy(self):
        return self.current_strategy
    
    def get_goal(self):
        return self.current_strategic_goal

def run_ruleDqn_dispatchHeuri_experiment(config, case, seed, **kwargs):
    """
    改进的层次强化学习版本 - 最小改动但真正实现层次RL
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
    hierarchical_buffer = []
    learning_rates = []
    start_time = time.time()
    gamma = config.gamma

    for episode in range(config.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        strategic_success_rates = []
        
        # 重置时间抽象层
        time_abstraction = TimeAbstractionLayer(config)
        
        while not done:
            # 层次决策：只在需要时调用Meta Agent
            if time_abstraction.should_decide():
                # 高层战略决策
                agent_idx, log_prob, entropy = meta_agent.select_action(state)
                
                # 推断战略目标（关键改进！）
                strategic_goal = infer_strategic_goal(agent_idx, state)
                
                # 存储策略和战略目标
                time_abstraction.set_strategy_and_goal(agent_idx, strategic_goal)
            else:
                # 使用缓存的策略和战略目标
                agent_idx = time_abstraction.get_strategy()
                strategic_goal = time_abstraction.get_goal()
                log_prob = torch.tensor(0.0)
            
            # 底层执行（保持不变，但现在底层知道战略意图）
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
            if agent_idx == 0 and rule_idx is not None:
                rule_dqn_agent.update({
                    'states': [state],
                    'actions': [rule_idx],
                    'rewards': [reward],
                    'next_states': [next_state],
                    'dones': [done]
                })
            
            # 计算基于战略目标的奖励（关键改进！）
            intrinsic_reward = calculate_goal_based_reward(
                strategic_goal, state, next_state, reward
            )
            
            # 存储层次经验
            hierarchical_buffer.append({
                'state': state,
                'agent_idx': agent_idx,
                'log_prob': log_prob,
                'env_reward': reward,
                'intrinsic_reward': intrinsic_reward,
                'strategic_goal': strategic_goal,  # 改为战略目标
                'step_type': 'meta' if time_abstraction.steps_since_last_decision == 0 else 'execution'
            })
            
            # 评估战略成功度
            success_rate = evaluate_strategic_success(strategic_goal, state, next_state)
            strategic_success_rates.append(success_rate)
            
            state = next_state
            episode_reward += reward

        # 计算层次回报
        hierarchical_returns = calculate_hierarchical_returns(hierarchical_buffer, gamma)
        
        # Meta Agent批量更新
        if len(hierarchical_buffer) >= config.batch_size:
            batch_indices = np.random.choice(
                len(hierarchical_buffer), 
                min(config.batch_size, len(hierarchical_buffer)), 
                replace=False
            )
            
            batch_data = [hierarchical_buffer[i] for i in batch_indices]
            
            # 确保所有数据都是有效的
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
            
            if len(valid_actions) >= 2:
                batch = {
                    'states': valid_states,
                    'actions': valid_actions,
                    'log_probs': valid_log_probs,
                    'returns': torch.tensor(valid_returns, dtype=torch.float32)
                }
                
                meta_loss = meta_agent.update(batch, optimizer, scheduler)
                stats['meta_losses'].append(meta_loss)
            else:
                meta_loss = 0.0
                print(f"Episode {episode}: Skipping meta update, only {len(valid_actions)} valid samples")
            
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            
            # 清理缓冲区
            keep_ratio = 0.3
            keep_size = int(len(hierarchical_buffer) * keep_ratio)
            hierarchical_buffer = hierarchical_buffer[-keep_size:]
        
        # 记录统计信息
        stats['episode_rewards'].append(episode_reward)
        stats['episode_lengths'].append(len(hierarchical_buffer))
        stats['makespans'].append(env.t)
        stats['tardy_penalty'].append(env.tardy_penalty)
        stats['total_tardiness'].append(env.total_weighted_tardiness)
        stats['machine_utilization'].append(calculate_machine_utilization(env))
        stats['strategic_success_rate'].append(np.mean(strategic_success_rates) if strategic_success_rates else 0)
        
        # 进度输出
        if episode % 10 == 0:
            current_lr = learning_rates[-1] if learning_rates else 0.0003
            success_rate = stats['strategic_success_rate'][-1]
            meta_updates = len(stats['meta_losses'])
            print(f'Episode {episode}, Reward: {episode_reward:.2f}, '
                  f'Strategic Success: {success_rate:.3f}, LR: {current_lr:.6f}, '
                  f'Meta Updates: {meta_updates}')

    # 保存结果
    total_time = time.time() - start_time
    additional_metrics = {
        "algorithm_type": "True_Hierarchical_RL_Production_Delivery",
        "batch_size": config.batch_size,
        "gamma": gamma,
        "final_learning_rate": learning_rates[-1] if learning_rates else 0.0003,
        "meta_decision_interval": time_abstraction.decision_interval,
        "total_meta_updates": len(stats['meta_losses'])
    }
    
    save_algorithm_results_csv(
        algo_name="True_Hierarchical_RL_Production+Delivery",
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
    total_work_time = sum(m.total_busy_time for m in env.machines)
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
