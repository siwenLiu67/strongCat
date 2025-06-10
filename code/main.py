import torch
import os
import numpy as np
from datetime import datetime
import logging 
import json
from algo.hdrl_framework import MetaController, create_dispatching_features
import algo.SchedulingPolicy as SchedulingPolicy
import algo.DispatchingPolicy as DispatchingPolicy
from entity.environment import WarehouseEnvironment
from data.outputPrinter import generate_comparison_plots

from matplotlib import pyplot as plt
from data.caseBuilder import Config

def main():
    # 1. 读取参数
    config = Config()
    
    # 2. 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # 3. 设置日志
    logger = logging.getLogger("WarehouseExperiment")
    logger.info("开始实验...")
    
    # 4. 设置随机种子
    torch.manual_seed(config['random_seed'])
    
    # 5. 初始化环境和智能体
    env = WarehouseEnvironment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化网络
    meta_controller = MetaController(
        state_dim=config['meta_state_dim'],
        action_dim=config['meta_action_dim'],
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    scheduling_policy = SchedulingPolicy(config).to(device)
    dispatching_policy = DispatchingPolicy(config).to(device)
    
    # 初始化优化器
    meta_optimizer = torch.optim.Adam(meta_controller.parameters(), lr=config['learning_rate'])
    scheduling_optimizer = torch.optim.Adam(scheduling_policy.parameters(), lr=config['learning_rate'])
    dispatching_optimizer = torch.optim.Adam(dispatching_policy.parameters(), lr=config['learning_rate'])
    
    # 6. 训练循环
    epsilon = config['epsilon_start']
    best_reward = float('-inf')
    
    for episode in range(config['num_episodes']):
        total_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            # 元控制器决策
            meta_state = torch.FloatTensor(state['meta']).unsqueeze(0).to(device)
            if np.random.random() < epsilon:
                meta_action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    meta_action = meta_controller.predict(meta_state).item()
            
            # 根据元动作选择子策略
            if meta_action == 0:  # 调度决策
                scheduling_features = create_scheduling_features(state, config)
                scheduling_action = scheduling_policy(scheduling_features)
                action = {'type': 'scheduling', 'action': scheduling_action}
            else:  # 配送决策
                dispatching_features = create_dispatching_features(state, config)
                dispatching_action = dispatching_policy(dispatching_features)
                action = {'type': 'dispatching', 'action': dispatching_action}
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # 存储经验
            # (这里需要实现经验回放缓冲区)

            
            # 更新状态
            state = next_state
        
        # 更新探索率
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        
        # 记录日志
        logger.info(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
        
        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'config': config,
                'episode': episode,
                'reward': best_reward
            }, f"{exp_dir}/best_model.pth")
        
        # 定期保存检查点
        if episode % 100 == 0:
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'meta_optimizer': meta_optimizer.state_dict(),
                'scheduling_optimizer': scheduling_optimizer.state_dict(),
                'dispatching_optimizer': dispatching_optimizer.state_dict(),
                'config': config,
                'episode': episode,
                'epsilon': epsilon
            }, f"{exp_dir}/checkpoint_ep{episode}.pth")
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main()


def visualize_progress(results, current_episode):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Reward plot
    plt.subplot(1, 3, 1)
    plt.plot(results['episode'], results['reward'])
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Tardiness plot
    plt.subplot(1, 3, 2)
    plt.plot(results['episode'], results['tardiness'])
    plt.title('Average Tardiness')
    plt.xlabel('Episode')
    plt.ylabel('Tardiness (hours)')
    
    # Utilization plot
    plt.subplot(1, 3, 3)
    plt.plot(results['episode'], results['utilization'])
    plt.title('Resource Utilization')
    plt.xlabel('Episode')
    plt.ylabel('Utilization (%)')
    
    plt.tight_layout()
    plt.savefig(f'results/progress_{current_episode}.png')
    plt.close()


def final_evaluation(agent, env, config):
    """Run final evaluation on test scenarios"""
    test_results = []
    for _ in range(10):  # 10 test scenarios
        state = env.reset()
        done = False
        while not done:
            actions = agent.get_actions(state, eval_mode=True)
            state, _, done, info = env.step(actions)
        test_results.append(info)
    
    # Save evaluation metrics
    with open('results/final_evaluation.json', 'w') as f:
        json.dump(test_results, f)
        
    # Generate comparison plots
    generate_comparison_plots(test_results)

def save_final_results(agent, results):
    """Save final models and results"""
    torch.save(agent.state_dict(), 'results/final_model.pt')
    
    with open('results/final_training_metrics.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
