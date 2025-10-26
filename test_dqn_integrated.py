import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 输入你的奖励值序列
rewards =[-798.3973192104463, -726.1439418072823, -1365.0782452317412, -651.4371078728043, -710.295599416037, 0.0, 0.0, -570.8651217527884, -764.0185519607954, -777.1054580184816, -758.7571460246787, -818.7682448014025, -975.8328552653024, -1382.7379967430643, -1546.8670708052805, 0.0, -664.7218738441361, -951.4370637050523, -1501.8295102057104, 0.0, 0.0, -853.1363356505614, 20.0, -483.6618222580254, -538.3763508487027, -673.6615852679054, -722.2337481189307, -877.017121592449, -897.2980417094911, -833.478002982786, -650.9764605864405, -610.7999417057019, -702.1977699785422, -770.0399663849325, -532.148350079734, -721.2712961344668, -269.29771296206536, -463.1481761068839, -940.8544957647521, -721.410708184319, -758.0549647241749, -481.1967992783817, 0.0, -324.74571648713265, -318.40445863052327, -709.7564357722599, -162.5984655603885, -461.6654718195803, -335.4201988576873, 0.0] 

# 2. 计算滑动均值（窗口大小10，平滑收敛趋势）
window_size = 10
steps = np.arange(1, len(rewards) + 1)
smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
# 调整平滑后的数据长度，与原始步数对齐（前window_size-1步用原始值填充）
smoothed_steps = steps[window_size-1:]
fill_values = rewards[:window_size-1]  # 前9步用原始值
full_smoothed = np.concatenate([fill_values, smoothed_rewards])

# 3. 绘制收敛曲线
fig, ax = plt.subplots(figsize=(12, 6))

# 绘制原始奖励值（灰色细点，展示波动）
ax.scatter(steps, rewards, color='#94a3b8', alpha=0.6, s=15, label='原始奖励值')

# 绘制滑动均值（蓝色实线，展示收敛趋势）
ax.plot(steps, full_smoothed, color='#2563eb', linewidth=2.5, label=f'滑动均值（窗口={window_size}）')

# 绘制参考线（奖励值=0，区分正负）
ax.axhline(y=0, color='#ef4444', linestyle='--', alpha=0.8, label='奖励值=0（参考线）')

# 设置图表标签和样式
ax.set_xlabel('训练步数', fontsize=12, fontweight='bold')
ax.set_ylabel('奖励值', fontsize=12, fontweight='bold')
ax.set_title('强化学习奖励值收敛曲线', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.legend()    
plt.tight_layout()
plt.show()