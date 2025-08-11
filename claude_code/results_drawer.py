import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv(r"f:\research_lsw\results\algorithm_comparison.csv")

# 只保留有效数据（排除num_jobs为0的算例）
df = df[df['num_jobs'] > 0]

# ----------- 原有 objective value 柱状图 -----------
grouped = df.groupby(['instance_id', 'algo_name'])['objective_value'].mean().reset_index()
instances = grouped['instance_id'].unique()
n_instances = len(instances)
ncols = 3
nrows = (n_instances + ncols - 1) // ncols
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), squeeze=False)

for idx, instance in enumerate(instances):
    row, col = divmod(idx, ncols)
    ax = axes[row][col]
    data = grouped[grouped['instance_id'] == instance].fillna(0)
    ax.bar(range(len(data['algo_name'])), data['objective_value'], color='skyblue')
    ax.set_title(f"{instance}")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Objective Value")
    ax.set_xticks(range(len(data['algo_name'])))
    ax.set_xticklabels(data['algo_name'], rotation=30)
    for i, v in enumerate(data['objective_value']):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

for idx in range(len(instances), nrows * ncols):
    row, col = divmod(idx, ncols)
    fig.delaxes(axes[row][col])

plt.tight_layout()
plt.show()

# ----------- 新增收敛曲线图 -----------
# 每个算例，每个算法，绘制平均收敛曲线
fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), squeeze=False)

for idx, instance in enumerate(instances):
    row, col = divmod(idx, ncols)
    ax = axes2[row][col]
    data = df[df['instance_id'] == instance]
    algo_names = data['algo_name'].unique()
    for algo in algo_names:
        rewards_list = []
        for rewards_str in data[data['algo_name'] == algo]['episode_rewards_series']:
            rewards = [float(r) for r in str(rewards_str).split(',') if r.strip() != '']
            rewards_list.append(rewards)
        if rewards_list:
            max_len = max(len(r) for r in rewards_list)
            rewards_arr = []
            for r in rewards_list:
                if len(r) < max_len:
                    r = r + [np.nan] * (max_len - len(r))
                rewards_arr.append(r)
            rewards_mean = np.nanmean(rewards_arr, axis=0)
            ax.plot(rewards_mean, label=algo)
    ax.set_title(f"{instance} 收敛曲线")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)

for idx in range(len(instances), nrows * ncols):
    row, col = divmod(idx, ncols)
    fig2.delaxes(axes2[row][col])

plt.tight_layout()
plt.show()