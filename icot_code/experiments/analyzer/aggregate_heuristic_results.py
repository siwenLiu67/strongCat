import pandas as pd
import re
import os

# 读取启发式运行结果CSV
input_path = os.path.join(os.path.dirname(__file__), '../../../results/heuristic_comparison_results.csv')
df = pd.read_csv(input_path)

def extract_m_n(instance_name):
    """
    从文件名中提取机器数m和工件数n，例如 instance_m10_j20_s1.json -> (10, 20)
    """
    match = re.search(r'm(\d+)_j(\d+)', instance_name)
    if match:
        m = int(match.group(1))
        n = int(match.group(2))
        return m, n
    return None, None

# 提取m和n
mn = df['instance'].apply(extract_m_n)
df['m'] = mn.apply(lambda x: x[0])
df['n'] = mn.apply(lambda x: x[1])

# 按算法、m、n分组求均值
agg_cols = ['makespan', 'total_cost', 'production_cost', 'transportation_cost', 'computation_time', 'schedule_length']
grouped = df.groupby(['algorithm', 'm', 'n'])[agg_cols].mean().reset_index()

# 输出到analyzer目录下
output_path = os.path.join(os.path.dirname(__file__), 'heuristic_grouped_mean.csv')
grouped.to_csv(output_path, index=False)

print(f"分组均值已保存到: {output_path}")
