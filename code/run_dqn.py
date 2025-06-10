import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List

from strongCat.code.algo.meta_controller import MetaController
from strongCat.code.algo.scheduling_policy import SchedulingPolicy 
from strongCat.code.algo.dispatching_policy import DispatchingPolicy
from strongCat.code.algo.feature_builder import FeatureBuilder
from strongCat.code.data.caseBuilder.config import Config
from strongCat.code.data.caseBuilder.jobshop_case_generator import FlexibleJobShopScenario
from strongCat.code.data.outputPrinter.training_period import TrainingVisualizer
from strongCat.code.entity.dynamic_fjsp_env import WarehouseEnvironment

def setup_logger(exp_dir: Path) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger('FJSP-DQN')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(exp_dir / 'training.log')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器 
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def run_episode(
    env: WarehouseEnvironment,
    meta_controller: MetaController,
    scheduling_policy: SchedulingPolicy,
    dispatching_policy: DispatchingPolicy,
    feature_builder: FeatureBuilder,
    epsilon: float,
    training: bool = True
) -> Dict:
    """运行一个训练回合"""
    state = env.reset()
    total_reward = 0
    episode_data = []
    done = False
    
    while not done:
        # 获取元控制器特征
        meta_features = feature_builder.create_meta_features(state)
        meta_features = torch.FloatTensor(meta_features).unsqueeze(0)
        
        # 元控制器选择动作
        meta_action, meta_q = meta_controller.act(meta_features, epsilon)
        
        # 根据元动作选择子策略
        if meta_action.item() == 0:  # 调度决策
            features = feature_builder.create_scheduling_features(state)
            action, log_prob = scheduling_policy.act(
                features['job_features'],
                features['machine_features'],
                features['job_adj'],
                features['machine_adj']
            )
            sub_action = {'type': 'scheduling', 'action': action}
        else:  # 配送决策
            features = feature_builder.create_dispatching_features(state)
            action, log_prob = dispatching_policy.act(
                features['job_features'],
                features['batch_features'],
                features['valid_mask']
            )
            sub_action = {'type': 'dispatching', 'action': action}
            
        # 执行动作
        next_state, reward, done, info = env.step(sub_action)
        total_reward += reward
        
        # 存储经验
        if training:
            episode_data.append({
                'state': state,
                'meta_action': meta_action,
                'sub_action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'meta_q': meta_q,
                'log_prob': log_prob
            })
            
        state = next_state
        
    return {
        'total_reward': total_reward,
        'episode_data': episode_data,
        'episode_length': len(episode_data)
    }

def main():
    # 1. 初始化配置
    config = Config()
    
    # 2. 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/fjsp_dqn_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 设置日志和可视化
    logger = setup_logger(exp_dir)
    visualizer = TrainingVisualizer(exp_dir)
    
    # 4. 生成训练案例
    case_generator = FlexibleJobShopScenario(config)
    train_cases = case_generator
    
    # 5. 初始化环境和网络
    env = WarehouseEnvironment(config, case=train_cases)
    feature_builder = FeatureBuilder(config)
    
    meta_controller = MetaController(config)
    scheduling_policy = SchedulingPolicy(config)
    dispatching_policy = DispatchingPolicy(config)
    
    # 6. 训练循环
    training_stats = {
        'episode_rewards': [],
        'meta_losses': [],
        'scheduling_losses': [],
        'dispatching_losses': [],
        'meta_decisions': []
    }
    
    best_reward = float('-inf')
    
    for episode in range(config.training.max_episodes):
        # 计算探索率
        epsilon = max(
            config.training.final_epsilon,
            config.training.initial_epsilon * 
            (config.training.epsilon_decay ** episode)
        )
        
        # 运行一个回合
        episode_info = run_episode(
            env, 
            meta_controller,
            scheduling_policy,
            dispatching_policy,
            feature_builder,
            epsilon,
            training=True
        )
        
        # 更新网络
        if episode >= config.training.warmup_episodes:
            meta_info = meta_controller.update(episode_info['episode_data'])
            sched_info = scheduling_policy.update(episode_info['episode_data'])
            disp_info = dispatching_policy.update(episode_info['episode_data'])
            
            training_stats['meta_losses'].append(meta_info)
            training_stats['scheduling_losses'].append(sched_info)
            training_stats['dispatching_losses'].append(disp_info)
        
        # 记录训练数据
        training_stats['episode_rewards'].append(episode_info['total_reward'])
        training_stats['meta_decisions'].extend(
            [{'action': d['meta_action']} for d in episode_info['episode_data']]
        )
        
        # 保存最佳模型
        if episode_info['total_reward'] > best_reward:
            best_reward = episode_info['total_reward']
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'config': config,
                'episode': episode,
                'reward': best_reward
            }, exp_dir / 'best_model.pt')
        
        # 可视化和日志
        if episode % config.training.log_interval == 0:
            visualizer.plot_rewards(training_stats['episode_rewards'], episode)
            visualizer.plot_losses(training_stats, episode)
            visualizer.plot_decision_distribution(training_stats, episode)
            visualizer.save_statistics(training_stats, episode)
            
            logger.info(f"Episode {episode}/{config.training.max_episodes}")
            logger.info(f"Total Reward: {episode_info['total_reward']:.2f}")
            logger.info(f"Best Reward: {best_reward:.2f}")
            logger.info(f"Epsilon: {epsilon:.4f}")
            
        # 定期保存检查点
        if episode % config.training.save_interval == 0:
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'training_stats': training_stats,
                'config': config,
                'episode': episode
            }, exp_dir / f'checkpoint_ep{episode}.pt')
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()