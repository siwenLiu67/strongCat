import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class HighLevelAgent(nn.Module):
    def __init__(self, config):
        super(HighLevelAgent, self).__init__()
        network_config = config.network.meta_controller
        
        # 确保输入维度与特征维度匹配(当前meta特征是5维)
        self.input_dim = 5  # [新作业比例, 负载不平衡度, 紧急程度, 调度方案年龄, 系统压力]
        self.hidden_dim = network_config.hidden_dim
        self.output_dim = 3  # [调度, 发运, 等待]
        
        # 修改网络层维度
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 优化器和参数设置
        self.optimizer = torch.optim.Adam(self.parameters(), lr=network_config.lr)
        self.gamma = network_config.gamma
        
        # 存储训练数据
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        """前向传播
        Args:
            state (torch.Tensor): 形状为 [batch_size, input_dim] 的状态张量
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits

    def masked_softmax(self, logits, mask):
        mask = torch.BoolTensor(mask).to(logits.device)
        logits[~mask] = float('-inf')
        return F.softmax(logits, dim=-1)

    def act(self, state, action_mask):
        """选择动作
        Args:
            state (torch.Tensor): 已经处理好的状态张量，形状为 [batch_size, input_dim]
            action_mask (List[bool]): 动作掩码
        """
        # 移除多余的维度转换，因为输入已经是正确形状
        logits = self.forward(state).squeeze(0)  # shape: [3]
        probs = self.masked_softmax(logits, action_mask)
        
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        loss = torch.tensor(0.0, dtype=torch.float32)
        for log_prob, R in zip(self.log_probs, returns):
            loss = loss - log_prob * R  # REINFORCE loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear storage
        self.log_probs = []
        self.rewards = []

    def update_from_batch(self, batch):
        # batch: dict with keys 'state', 'action', 'reward', all tensors
        states = batch['state'].float()
        actions = batch['action'].long()
        rewards = batch['reward'].float()

        # Normalize rewards
        if rewards.size(0) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = -torch.mean(chosen_log_probs * rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
