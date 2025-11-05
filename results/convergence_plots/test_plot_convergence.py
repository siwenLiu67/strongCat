import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional

def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    计算滑动平均值
    
    Args:
        data: 输入数据数组
        window_size: 滑动窗口大小
    
    Returns:
        滑动平均值数组
    """
    if window_size <= 0:
        raise ValueError("窗口大小必须为正整数")
    
    if window_size > len(data):
        raise ValueError("窗口大小不能超过数据长度")
    
    moving_avg = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        moving_avg.append(sum(window) / window_size)
    
    return moving_avg

def plot_moving_average_convergence(
    data: List[float],
    window_size: int = 10,
    title: str = "滑动平均收敛图",
    xlabel: str = "迭代次数",
    ylabel: str = "数值",
    data_label: str = "原始数据",
    ma_label: str = "滑动平均",
    figsize: tuple = (12, 6),
    show_original: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    grid: bool = True,
    alpha: float = 0.7
):
    """
    绘制滑动平均收敛图
    
    Args:
        data: 输入数据数组
        window_size: 滑动窗口大小
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        data_label: 原始数据标签
        ma_label: 滑动平均数据标签
        figsize: 图表尺寸
        show_original: 是否显示原始数据
        save_path: 保存路径，如果为None则不保存
        dpi: 图片分辨率
        grid: 是否显示网格
        alpha: 透明度
    """
    # 计算滑动平均
    ma_data = moving_average(data, window_size)
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制原始数据（如果要求显示）
    if show_original:
        plt.plot(data, label=data_label, alpha=alpha*0.8, color='blue', linewidth=1)
    
    # 绘制滑动平均数据
    ma_x = range(window_size - 1, len(data))
    plt.plot(ma_x, ma_data, label=ma_label, color='red', linewidth=2)
    
    # 设置图表属性
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    # 设置紧凑布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    # 显示图表
    plt.show()
    
    return ma_data

def plot_multiple_moving_average(
    data_list: List[List[float]],
    labels: List[str],
    window_size: int = 10,
    title: str = "多组数据滑动平均收敛图",
    xlabel: str = "迭代次数",
    ylabel: str = "数值",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None
):
    """
    绘制多组数据的滑动平均收敛图
    
    Args:
        data_list: 多组数据列表
        labels: 每组数据的标签
        window_size: 滑动窗口大小
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图表尺寸
        save_path: 保存路径
        colors: 颜色列表
    """
    if len(data_list) != len(labels):
        raise ValueError("数据列表和标签列表长度必须相同")
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=figsize)
    
    for i, (data, label) in enumerate(zip(data_list, labels)):
        ma_data = moving_average(data, window_size)
        ma_x = range(window_size - 1, len(data))
        color = colors[i % len(colors)]
        
        plt.plot(ma_x, ma_data, label=label, color=color, linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 示例1：单组数据
    print("示例1：单组数据滑动平均收敛图")
    
    rewards = [-1262.1978,-2570.6878,-1995.2456,-2852.1343,-2520.4809,-2347.8113,-4741.5988,-3042.3000,-3099.0611,-1041.4710,-2366.5471,-2081.8863,-2084.5974,-3249.0263,-1842.9648,-935.5967,-1437.2383,-2145.9916,-2539.8522,-2109.4995,-2056.3379,-2611.7573,-2478.0552,-2298.3230,-2176.9517,-954.3685,-2220.9203,-2170.3631,-2342.0297,-2061.2672,-2463.9792,-2089.6173,-2598.9393,-1123.4077,-1539.6983,-1611.2543,-1461.4720,-2071.9116,-2396.8598,-3223.8585,-2021.6427,-2665.7529,-2219.4514,-2413.2137,-1922.1497,-2297.7725,-1478.2060,-2594.6771,-2405.9410,-2077.9600,-1813.7766,-2432.8908,-2743.1660,-1803.0149,-2326.9839,-2283.4041,-2437.4286,-2369.7996,-2451.4266,-1939.8883,-1695.9530,-1357.3024,-1220.2595,-1917.8384,-765.4972,-562.5594,-1205.6716,-2197.5817,-1184.2366,-975.9580,-2372.0038,-2363.0377,-1737.7873,-2696.2678,-1853.5418,-1146.0614,-2227.9474,-3025.3645,-1847.3207,-2116.5447,-1054.1996,-863.8321,-2555.4850,-1068.8935,-2319.8574,-2363.0766,-1547.5248,-1844.6191,-2159.7155,-2303.1663,-2255.2668,-2170.7586,-1826.4739,-2034.5294,-1977.9781,-944.6343,-872.8944,-1404.8560,-2173.2893,-1673.3774,-1178.0969,-745.0403,-771.6750,-1533.2954,-1544.0232,-2239.1397,-2736.4836,-1731.7268,-1465.7390,-2711.8337,-1188.6387,-2141.8897,-1877.8261,-1799.9152,-1629.5874,-2332.8768,-1584.1860,-713.9253,-2000.7217,-1583.3171,-796.8075,-803.8468,-1368.2907,-1066.5850,-1126.4012,-2596.8438,-2522.1947,-1525.5533,-2028.1944,-2492.6649,-2123.4231,-1907.7303,-702.9128,-862.2498,-2255.7718,-1116.2102,-686.2402,-602.6305,-849.6228,-1566.2196,-650.0869,-2282.6666,-1271.7025,-1878.8371,-1459.3954,-2328.4203,-994.2320,-1029.6521,-1155.4161,-1506.0366,-1094.8731,-1991.2069,-736.6384,-589.3441,-1045.9792,-1405.8716,-574.6323,-887.0642,-469.8470,-1372.3682,-1247.9592,-1482.4322,-896.1409,-1405.7932,-1483.8600,-1930.4546,-678.3718,-748.8211,-704.4707,-800.1586,-830.9070,-630.9043,-962.7944,-1524.3396,-1235.5114,-889.6731,-1055.9620,-1202.9871,-1239.8888,-1322.6970,-1040.1661,-1030.6743,-992.9603,-959.5500,-1137.6053,-1636.8420,-1172.2166,-612.1822,-989.9323,-1136.0426,-1207.5443,-826.7778,-729.2013,-1365.8059,-1415.7782,-1414.1002,-868.7825,-1309.2399,-576.4776,-1061.7076]
    
    # 绘制滑动平均收敛图
    ma_rewards = plot_moving_average_convergence(
        data=rewards,
        window_size=20,
        title="强化学习奖励收敛图",
        xlabel="训练回合",
        ylabel="奖励值",
        data_label="原始奖励",
        ma_label="滑动平均 (窗口=20)",
        save_path="reward_convergence.png"  # 可选：保存图片
    )
    
    # # 示例2：多组数据对比
    # print("\n示例2：多组数据对比")
    
    # # 生成多组示例数据
    # data_groups = []
    # labels = ["算法A", "算法B", "算法C"]
    
    # for label in labels:
    #     np.random.seed(hash(label) % 1000)
    #     episodes = 150
    #     rewards = []
    #     current_reward = 0
        
    #     for i in range(episodes):
    #         improvement = np.random.normal(1.2, 0.3)
    #         current_reward += improvement
    #         noise = np.random.normal(0, 1.5)
    #         rewards.append(max(0, current_reward + noise))
        
    #     data_groups.append(rewards)
    
    # # 绘制多组数据对比图
    # plot_multiple_moving_average(
    #     data_list=data_groups,
    #     labels=labels,
    #     window_size=15,
    #     title="不同算法奖励收敛对比",
    #     xlabel="训练回合",
    #     ylabel="奖励值",
    #     save_path="algorithm_comparison.png"  # 可选：保存图片
    # )
    
    # print("示例运行完成！")

