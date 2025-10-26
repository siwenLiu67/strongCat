import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 输入你的奖励值序列
rewards =[-3176.580314821757, -2436.6363594455456, -2328.3370221083514, -3951.766473731497, -3452.452844952782, -2748.499212112824, -2110.775603195198, -3834.3133411785966, -2195.5751882840304, 0.0, -3787.6765266693023, -3685.818003432443, -3330.0830291524303, -2968.217095698002, -3857.9978547955375, 20.0, -2172.9774076530116, -1437.027423259889, -873.3430700192964, -1133.5897422094586, -1803.2788607878365, -1095.7615034471123, -1445.193481095158, -1372.4130162754236, -1938.4809158080725, -1475.6371108967003, -1632.2035614461317, -1048.1307366690535, -1845.726572240242, -1461.4407146682827, -1710.2267665743987, -828.4669826591905, -1339.7730639799865, -1699.5554140464064, -1681.0707597863309, -1394.6731599712186, -1329.8891417014588, -1230.5982751929546, -1174.564580808119, -1188.9257336337294, -1401.3772291366863, -1302.1077298794567, -1515.2055168388727, -848.7523253080196, -446.0481129222635, -804.9645651078552, -419.69147479791513, -1685.9065802006328, -916.327629868754, -197.44146178326412] 

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