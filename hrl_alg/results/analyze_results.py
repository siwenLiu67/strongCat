import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

all_rewards = results['all_rewards']
all_rewards_per_episode = results['all_rewards_per_episode']
all_dispatch_counts = results.get('all_dispatch_counts', [])
all_schedule_counts = results.get('all_schedule_counts', [])
all_steps = results.get('all_steps', [])

# 1. 每个episode累计奖励趋势
plt.figure(figsize=(6,4))
plt.plot(all_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.tight_layout()
plt.savefig('total_reward_per_episode.png')
plt.show()

# 2. 收敛曲线（平均累计奖励）
max_len = max(len(r) for r in all_rewards_per_episode)
rewards_matrix = np.zeros((len(all_rewards_per_episode), max_len))
for i, rewards in enumerate(all_rewards_per_episode):
    rewards_matrix[i, :len(rewards)] = np.cumsum(rewards)
mean_cumulative = rewards_matrix.mean(axis=0)
std_cumulative = rewards_matrix.std(axis=0)
plt.figure(figsize=(6,4))
plt.plot(mean_cumulative, color='black', linewidth=2, label='Mean Cumulative Reward')
plt.fill_between(range(len(mean_cumulative)), mean_cumulative-std_cumulative, mean_cumulative+std_cumulative, alpha=0.2)
plt.xlabel('Step')
plt.ylabel('Cumulative Reward')
plt.title('Convergence Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('convergence_curve.png')
plt.show()

# 3. 每步奖励均值和方差
step_means = rewards_matrix.mean(axis=0)
step_stds = rewards_matrix.std(axis=0)
plt.figure(figsize=(6,4))
plt.plot(step_means, label='Mean Reward per Step')
plt.fill_between(range(len(step_means)), step_means-step_stds, step_means+step_stds, alpha=0.2)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Mean and Std of Reward per Step')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('reward_per_step.png')
plt.show()

# 4. 调度/配送动作次数分布
if all_dispatch_counts and all_schedule_counts:
    plt.figure(figsize=(6,4))
    plt.plot(all_dispatch_counts, label='Dispatch Count', marker='o')
    plt.plot(all_schedule_counts, label='Schedule Count', marker='x')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.title('Dispatch & Schedule Count per Episode')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('action_counts.png')
    plt.show()

