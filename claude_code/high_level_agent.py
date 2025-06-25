import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import torch
from environment import WarehouseEnvironment

class HighLevelAgent:
    """元控制器
    
    负责在更高层次上做出决策:
    1. 是否进行调度
    2. 是否进行配送
    3. 是否等待观察
    """
    
    def __init__(self, config):
        """初始化元控制器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self._build_network()
        self._setup_training()

    def _build_features(self, state: Dict) -> torch.Tensor:
        """从环境中提取高层决策智能体的特征
            用于决定是否触发新的调度方案
        特征包含:
        1. q_new: 新到达作业占当前总作业数的比例
        2. sigma_mach_t: 机器负载不平衡程度
        3. urgency_ratio: 紧急程度
        4. schedule_age: 当前调度方案的年龄
        5. system_pressure: 系统压力指标
        
        Args:
            state: 环境状态
            
        Returns:
            
        """

        current_time = state['current_time']
        jobs = state['available_jobs']
        machines = state['machines']
        last_schedule_time = state.get('last_schedule_time', 0)
        
        # 1. 新到达作业比例
        new_jobs = len(state['available_jobs']) - len(state['completed_jobs'])
        q_new = new_jobs / max(1, len(jobs))
        
        # 2. 机器负载不平衡度
        remaining_times = [float(m.remaining_time) for m in machines]
        sigma_mach_t = torch.tensor(remaining_times, dtype=torch.float32).std().item() / self.config.max_processing_time
        
        # 3. 紧急程度
        urgent_threshold = self.config.urgent_threshold
        urgency_ratio = 0
        
        # 4. 当前调度方案的年龄
        schedule_age = self.normalize_time(current_time - last_schedule_time)
        
        # 5. 系统压力指标
        # 考虑剩余加工容量与待加工工作量的比例
        system_pressure = self._calculate_system_pressure(state)

        meta_features = torch.tensor([
            q_new,           # 新作业比例
            sigma_mach_t,    # 负载不平衡度
            urgency_ratio,   # 紧急程度
            schedule_age,    # 调度方案年龄
            system_pressure  # 系统压力
        ], dtype=torch.float32)
        
        return meta_features
        
    def _calculate_system_pressure(self, Dict):
        return 1

    def normalize_time(self, time_value):
        """归一化时间值到[0, 1]区间"""
        max_time = self.config.max_processing_time
        return float(time_value) / max(1, max_time)

    def _build_network(self):
        """构建简单的MLP策略网络"""
        input_dim = 5  # 特征数量，和_build_features输出一致
        hidden_dim = 32
        output_dim = 3  # 动作数：调度、配送、等待
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _setup_training(self):
        """设置优化器等训练参数"""
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)


    def select_action(self, state: Dict):
        """选择动作
        
        Args:
            state: 环境状态
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        # 提取特征
        features = self._build_features(state)
        logits = self.policy_net(features)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
        
    def update(self, batch: Dict):
        """更新策略网络
        
        Args:
            batch: 经验数据批次
            
        Returns:
            loss: 损失值
        """
        # TODO: 实现网络更新
        pass

    

    