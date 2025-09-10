import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取CSV文件
df = pd.read_csv('results/all_experiment_results.csv')

# 过滤出包含reward_curve字段的行
def has_reward_curve(extra_info):
    if pd.isna(extra_info) or extra_info == '' or extra_info == '{}':
        return False
    try:
        info_dict = ast.literal_eval(extra_info)
        return 'reward_curve' in info_dict
    except:
        return False

# 应用过滤
filtered_df = df[df['extra_info'].apply(has_reward_curve)].copy()

# 解析reward_curve数据
def extract_reward_curve(extra_info):
    try:
        info_dict = ast.literal_eval(extra_info)
        return info_dict.get('reward_curve', [])
    except:
        return []

filtered_df['reward_curve'] = filtered_df['extra_info'].apply(extract_reward_curve)

# 解析metrics数据以获取总成本等信息
def parse_metrics(metrics_str):
    try:
        metrics_dict = ast.literal_eval(metrics_str)
        return metrics_dict.get('total_cost', np.nan)
    except:
        return np.nan

filtered_df['total_cost'] = filtered_df['metrics'].apply(parse_metrics)

print(f"找到 {len(filtered_df)} 条包含reward_curve数据的记录")
print("\n算法分布:")
print(filtered_df['algorithm_name'].value_counts())
print("\n实例分布:")
print(filtered_df['instance_id'].value_counts())

# 从instance_id中提取机器数和工件数信息
def extract_instance_info(instance_id):
    # 实例ID格式: instance_m{机器数}_j{工件数}_s{seed}
    parts = instance_id.split('_')
    machines = int(parts[1][1:])  # 去掉'm'前缀
    jobs = int(parts[2][1:])      # 去掉'j'前缀
    seed = int(parts[3][1:])      # 去掉's'前缀
    return machines, jobs, seed

filtered_df['machines'] = filtered_df['instance_id'].apply(lambda x: extract_instance_info(x)[0])
filtered_df['jobs'] = filtered_df['instance_id'].apply(lambda x: extract_instance_info(x)[1])
filtered_df['seed'] = filtered_df['instance_id'].apply(lambda x: extract_instance_info(x)[2])

print("\n规模分布:")
print(filtered_df.groupby(['machines', 'jobs']).size())

# 按规模（机器数和工件数）和算法分组绘制收敛曲线
scales = filtered_df[['machines', 'jobs']].drop_duplicates().values.tolist()
algorithms = filtered_df['algorithm_name'].unique()

# 创建输出目录
os.makedirs('results/convergence_plots', exist_ok=True)

# 为每个规模创建单独的图表
for scale in scales:
    machines, jobs = scale
    scale_data = filtered_df[(filtered_df['machines'] == machines) & (filtered_df['jobs'] == jobs)]
    
    if len(scale_data) == 0:
        continue
        
    # 用中文格式
    rcParams = {'font.family': 'sans-serif',
                'font.sans-serif': ['SimHei'], # 指定默认字体
                'axes.unicode_minus': False} # 解决负号'-'显示为方块的问题
    plt.rcParams.update(rcParams)
    plt.figure(figsize=(12, 8))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, algo in enumerate(algorithms):
        algo_data = scale_data[scale_data['algorithm_name'] == algo]
        
        if len(algo_data) > 0:
            # 对同一规模不同seed的reward_curve取平均值
            all_curves = []
            final_costs = []
            
            for _, row in algo_data.iterrows():
                if row['reward_curve']:
                    all_curves.append(row['reward_curve'])
                    final_costs.append(row['total_cost'])
            
            if all_curves:
                # 找到最长的曲线长度
                max_length = max(len(curve) for curve in all_curves)
                
                # 填充较短的曲线为NaN
                padded_curves = []
                for curve in all_curves:
                    if len(curve) < max_length:
                        padded_curve = curve + [np.nan] * (max_length - len(curve))
                    else:
                        padded_curve = curve
                    padded_curves.append(padded_curve)
                
                # 计算平均值
                mean_curve = np.nanmean(padded_curves, axis=0)
                avg_final_cost = np.mean(final_costs)
                
                episodes = range(1, len(mean_curve) + 1)
                plt.plot(episodes, mean_curve, 
                        label=f'{algo} (平均最终成本: {avg_final_cost:.2f})', 
                        color=colors[i], linewidth=2, marker='o')
    
    plt.xlabel('训练轮次')
    plt.ylabel('平均奖励值')
    plt.title(f'规模: {machines}台机器 × {jobs}个工件 - 算法收敛曲线对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(f'results/convergence_plots/m{machines}_j{jobs}_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已为规模 {machines}台机器 × {jobs}个工件 创建收敛曲线图")

# 创建所有实例的汇总图表
plt.figure(figsize=(15, 10))

for i, algo in enumerate(algorithms):
    algo_data = filtered_df[filtered_df['algorithm_name'] == algo]
    
    if len(algo_data) > 0:
        # 对每个算法的所有reward_curve取平均值
        all_curves = []
        for curve in algo_data['reward_curve']:
            if curve:
                all_curves.append(curve)
        
        if all_curves:
            # 找到最长的曲线长度
            max_length = max(len(curve) for curve in all_curves)
            
            # 填充较短的曲线为NaN
            padded_curves = []
            for curve in all_curves:
                if len(curve) < max_length:
                    padded_curve = curve + [np.nan] * (max_length - len(curve))
                else:
                    padded_curve = curve
                padded_curves.append(padded_curve)
            
            # 计算平均值和标准差
            mean_curve = np.nanmean(padded_curves, axis=0)
            std_curve = np.nanstd(padded_curves, axis=0)
            
            episodes = range(1, len(mean_curve) + 1)
            plt.plot(episodes, mean_curve, label=algo, linewidth=2)
            plt.fill_between(episodes, 
                           mean_curve - std_curve, 
                           mean_curve + std_curve, 
                           alpha=0.2)

plt.xlabel('训练轮次')
plt.ylabel('平均奖励值')
plt.title('所有实例 - 算法收敛曲线对比（平均值±标准差）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/convergence_plots/all_instances_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

print("已创建所有实例的汇总收敛曲线图")

# 输出统计信息
print("\n=== 统计信息 ===")
for algo in algorithms:
    algo_data = filtered_df[filtered_df['algorithm_name'] == algo]
    if len(algo_data) > 0:
        final_costs = algo_data['total_cost'].dropna()
        if len(final_costs) > 0:
            print(f"{algo}:")
            print(f"  记录数: {len(algo_data)}")
            print(f"  平均最终成本: {final_costs.mean():.2f}")
            print(f"  最终成本标准差: {final_costs.std():.2f}")
            print(f"  最小最终成本: {final_costs.min():.2f}")
            print(f"  最大最终成本: {final_costs.max():.2f}")
            print()

print("分析完成！图表已保存到 results/convergence_plots/ 目录")
