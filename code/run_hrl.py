import torch 
import os
import numpy as np 
from datetime import datetime
import logging
from collections import deque
import random
from typing import Dict, List
from data.outputPrinter.training_period import TrainingVisualizer
from algo.high_level_agent import HighLevelAgent
from algo.scheduling_policy import SchedulingPolicy
from algo.dispatching_policy import DispatchingPolicy
from algo.feature_builder import FeatureBuilder
from entity.dynamic_fjsp_env import WarehouseEnvironment
from entity.config import Config
from data.caseBuilder.jobshop_case_generator import FlexibleJobShopScenario

def setup_logger(exp_dir: str) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger("WarehouseExperiment")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(f"{exp_dir}/train.log")
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

class ExperienceBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Dict):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

def main():
    # 1. 读取配置
    config = Config()
    
    # 2. 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/exp_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 3. 设置日志
    logger = setup_logger(exp_dir)
    logger.info("开始实验...")
    
    # 4. 设置随机种子
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    # 生成算例
    case = FlexibleJobShopScenario(config)
    logger.info(f"生成算例: {case.summary()}")
    logger.info("算例生成完成")
    
    # 5. 初始化环境和特征构建器
    env = WarehouseEnvironment(config, case.get_case_info())
    feature_builder = FeatureBuilder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 6. 初始化策略网络
    meta_controller = HighLevelAgent(config).to(device)
    scheduling_policy = SchedulingPolicy(config).to(device)
    dispatching_policy = DispatchingPolicy(config).to(device)
    
    # 7. 初始化经验缓冲区
    meta_buffer = ExperienceBuffer(config.training.meta_controller.buffer_size)
    scheduling_buffer = ExperienceBuffer(config.training.scheduling.buffer_size)
    dispatching_buffer = ExperienceBuffer(config.training.dispatching.buffer_size)
    
    # 8. 训练循环
    total_steps = 0
    best_reward = float('-inf')
    training_stats = {
        'episode_rewards': [],
        'meta_losses': [],
        'scheduling_losses': [],
        'dispatching_losses': []
    }
    
    for episode in range(config.training.max_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 元控制器决策
            meta_features = feature_builder._build_meta_features(state)['meta_features']
        
            # 将特征转换为张量并移动到设备
            meta_features = torch.FloatTensor(meta_features).unsqueeze(0).to(device)
            action_mask = [True, np.random.rand() > 0.5, True]

            meta_output = meta_controller.act(meta_features, action_mask)
            meta_action = meta_output[0] if isinstance(meta_output, (list, tuple)) else meta_output
            meta_log_prob = meta_output[1] if isinstance(meta_output, (list, tuple)) else 0.0
            
            # 根据元动作选择子策略
            if meta_action == 0:  # 调度决策
                # 确保state包含必要的属性
                if not hasattr(state, 'jobs') or not hasattr(state, 'machines'):
                    raise ValueError("Invalid state: missing jobs or machines")
                
                # 构建特征前验证数据
                try:
                    features = feature_builder._build_scheduling_features(state)
                    job_features = features['job_features']
                    job_adj = features['job_adj']
                    machine_features = features['machine_features']
                    machine_adj = features['machine_adj']
                    
                    scheduling_action, sched_log_prob = scheduling_policy.act(
                        job_features, job_adj, machine_features, machine_adj
                    )
                    action = {'type': 'scheduling', 'action': scheduling_action}
                except Exception as e:
                    logger.error(f"Error building scheduling features: {str(e)}")
                    raise
            else:  # 配送决策
                # 将环境状态转换为字典格式
                state_dict = {
                    'jobs': state.jobs,
                    'batches': state.batches,
                    'current_time': state.current_time
                }
                job_features, batch_features, valid_mask = (
                    feature_builder._build_dispatching_features(state_dict)
                )
                dispatching_action, disp_log_prob = dispatching_policy.act(
                    job_features, batch_features, valid_mask
                )
                action = {'type': 'dispatching', 'action': dispatching_action}
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            total_steps += 1
            
            # 存储经验
            if meta_action == 0:
                scheduling_buffer.push({
                    'state': state,
                    'action': scheduling_action,
                    'reward': reward,
                    'next_state': next_state,
                    'log_prob': sched_log_prob,
                    'done': done
                })
            else:
                dispatching_buffer.push({
                    'state': state,
                    'action': dispatching_action,
                    'reward': reward,
                    'next_state': next_state,
                    'log_prob': disp_log_prob,
                    'done': done
                })
                
            meta_buffer.push({
                'state': state,
                'action': meta_action,
                'reward': reward,
                'next_state': next_state,
                'log_prob': meta_log_prob,
                'done': done
            })
            
            # 更新网络
            if total_steps % config.training.meta_controller.update_freq == 0:
                if len(meta_buffer) >= config.training.meta_controller.batch_size:
                    batch = meta_buffer.sample(config.training.meta_controller.batch_size)
                    # Convert list of dicts to dict of tensors
                    batch_dict = {k: torch.tensor([d[k] for d in batch]) if not isinstance(batch[0][k], torch.Tensor) else torch.stack([d[k] for d in batch]) for k in batch[0]}
                    meta_info = meta_controller.update_from_batch(batch_dict)
                    training_stats['meta_losses'].append(meta_info)
            
            if total_steps % config.training.scheduling.policy_update_freq == 0:
                if len(scheduling_buffer) >= config.training.scheduling.batch_size:
                    batch = scheduling_buffer.sample(config.training.scheduling.batch_size)
                    batch_dict = {k: torch.tensor([d[k] for d in batch]) if not isinstance(batch[0][k], torch.Tensor) else torch.stack([d[k] for d in batch]) for k in batch[0]}
                    sched_info = scheduling_policy.update(batch_dict)
                    training_stats['scheduling_losses'].append(sched_info)
            
            if total_steps % config.training.dispatching.policy_update_freq == 0:
                if len(dispatching_buffer) >= config.training.dispatching.batch_size:
                    batch = dispatching_buffer.sample(config.training.dispatching.batch_size)
                    batch_dict = {k: torch.tensor([d[k] for d in batch]) if not isinstance(batch[0][k], torch.Tensor) else torch.stack([d[k] for d in batch]) for k in batch[0]}
                    disp_info = dispatching_policy.update(batch_dict)
                    training_stats['dispatching_losses'].append(disp_info)
            
            state = next_state
        
        # 记录训练统计
        training_stats['episode_rewards'].append(episode_reward)
        
        # 日志记录
        logger.info(f"Episode {episode}/{config.training.max_episodes}")
        logger.info(f"Total Reward: {episode_reward:.2f}")
        logger.info(f"Average Loss - Meta: {np.mean(training_stats['meta_losses'][-10:]):.4f}")
        logger.info(f"Average Loss - Scheduling: {np.mean(training_stats['scheduling_losses'][-10:]):.4f}")
        logger.info(f"Average Loss - Dispatching: {np.mean(training_stats['dispatching_losses'][-10:]):.4f}")
        
        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            save_path = f"{exp_dir}/best_model.pt"
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'config': config,
                'episode': episode,
                'reward': best_reward
            }, save_path)
            
        # 定期保存检查点
        if episode % config.training.save_interval == 0:
            save_path = f"{exp_dir}/checkpoint_ep{episode}.pt"
            torch.save({
                'meta_controller': meta_controller.state_dict(),
                'scheduling_policy': scheduling_policy.state_dict(),
                'dispatching_policy': dispatching_policy.state_dict(),
                'training_stats': training_stats,
                'config': config,
                'episode': episode
            }, save_path)
        
        # 可视化训练进度
        if episode % config.training.log_interval == 0:
            visualizer = TrainingVisualizer(exp_dir)
            visualizer.plot_rewards(training_stats['episode_rewards'], episode)
            visualizer.plot_losses(training_stats, episode)
            visualizer.plot_decision_distribution(training_stats, episode)
    
    logger.info("训练完成!")
    
    # 最终评估
   # final_evaluation(meta_controller, scheduling_policy, dispatching_policy, env, config)
   # save_final_results(training_stats, exp_dir)

if __name__ == "__main__":
    main()
