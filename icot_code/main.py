# main.py

import sys
import os

# 将项目根目录添加到 sys.path，以便正确导入 config 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from data_loader import (
    load_production_data,
    load_transportation_data,
    load_drl_hyperparameters,
    load_search_parameters
)
from models.production_env import FJSPEnv
from models.transportation_model import plan_transportation
from agents.ppo_agent import PPOAgent, DRL_Scheduling_Agent

def define_search_space(prod_data, trans_data):
    """
    根据逻辑边界定义 T_internal 的搜索空间。
    """
    # 修正后的简化生产下界计算
    min_job_times = []
    for job_id, ops in prod_data['jobs'].items():
        job_time = 0
        for op_id, machine_times in ops:
            job_time += min(machine_times.values())
        min_job_times.append(job_time)

    # 生产下界 = 所有作业中理想加工时间最长的那个
    min_production_time = max(min_job_times)
    
    # 上界: 每个市场订单的最晚生产完成时间中的最小值
    max_production_times = []
    for order_id, order_details in trans_data['orders'].items():
        deadline = order_details['due']
        market = order_details['market']
        
        # 找到到这个市场的最快运输时间
        min_transport_time_market = float('inf')
        if market in trans_data['transport_data']:
            for mode, providers in trans_data['transport_data'][market].items():
                for provider, params in providers.items():
                    transport_time = params['distance'] / params['speed']
                    if transport_time < min_transport_time_market:
                        min_transport_time_market = transport_time
        
        if min_transport_time_market != float('inf'):
            max_production_times.append(deadline - min_transport_time_market)

    if not max_production_times:
        raise ValueError("无法为任何订单确定有效的生产时间上限。")

    max_production_time = min(max_production_times)

    return int(min_production_time), int(max_production_time)


def evaluate_t_internal(t_internal, prod_data, trans_data, drl_params, max_episodes):
    """
    为给定的 t_internal 评估总成本。
    """
    print(f"\n评估 T_internal = {t_internal}")
    
    # 为此 T_internal 初始化环境和代理
    env = FJSPEnv(prod_data, trans_data['orders'], t_internal)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim, drl_params)

    # --- 底层战术执行 ---
    # a. 用于生产调度的 DRL
    print(f"  - 开始 DRL 生产调度 (T_internal={t_internal}, 轮次={max_episodes})...")
    s_prod, c_max = DRL_Scheduling_Agent(env, agent, max_episodes)
    print(f"  - DRL 生产调度完成 (C_max={c_max}).")
    
    # 如果 DRL 代理未能满足 t_internal，则应用惩罚
    penalty_cost = 0
    if c_max > t_internal:
        print(f"  - DRL 代理未能满足 T_internal (C_max={c_max})。应用惩罚并使用 T_internal 进行运输规划。")
        penalty_cost = drl_params.get('penalty_factor', 100) * (c_max - t_internal)**2
        c_max_for_transport = t_internal
    else:
        c_max_for_transport = c_max

    # b. 确定性运输规划
    print(f"  - 开始确定性运输规划 (生产完成时间={c_max_for_transport})...")
    c_transport, s_trans = plan_transportation(c_max_for_transport, trans_data['transport_data'], trans_data['orders'])
    print(f"  - 运输规划完成 (运输成本={c_transport}).")

    if c_transport == float('inf'):
        print(f"  - 对于生产完成时间={c_max_for_transport} 没有可行的运输计划。返回无限大成本。")
        return t_internal, float('inf'), None, None

    # c. 评估总成本
    c_production = c_max * prod_data['c_unit_production']
    total_cost = c_production + c_transport + penalty_cost
    print(f"  - C_max: {c_max}, 生产成本: {c_production}, 运输成本: {c_transport}, 惩罚成本: {penalty_cost}, 总成本: {total_cost}")

    plan = {'production': s_prod, 'transport': s_trans}
    return t_internal, total_cost, plan, c_max

def plot_results(results):
    """
    将 T_internal 与总成本的关系可视化。
    """
    # 配置 matplotlib 以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    t_internals = [res[0] for res in results]
    total_costs = [res[1] for res in results if res[1] != float('inf')]
    
    if not total_costs:
        print("没有有效的解决方案可供绘图。")
        return

    plt.figure(figsize=(12, 7))
    plt.plot(t_internals, [res[1] if res[1] != float('inf') else max(total_costs) * 1.1 for res in results], 'bo-', label='总成本')
    
    best_t_internal = min(results, key=lambda x: x[1])[0]
    best_cost = min(total_costs)
    
    plt.axvline(x=best_t_internal, color='r', linestyle='--', label=f'最优 T_internal = {best_t_internal}')
    plt.scatter([best_t_internal], [best_cost], color='red', s=100, zorder=5, label=f'最低成本 = {best_cost:.2f}')
    
    plt.title('T_internal 对总成本的影响')
    plt.xlabel('T_internal (内部生产截止时间)')
    plt.ylabel('总成本')
    plt.legend()
    plt.grid(True)
    
    # 保存图表到文件
    plt.savefig('t_internal_vs_total_cost.png')
    print("\n结果图已保存到 't_internal_vs_total_cost.png'")

def main():
    """
    运行 HD-DRL 算法的主函数。
    """
    # 打印 PPO 代理将使用的设备
    try:
        from agents.ppo_agent import device
        print(f"DRL 代理将使用: {device}")
    except ImportError:
        print("无法导入 PPO 代理设备信息。")

    file_path = './instances/instance_m10_j20_s1.json'  # 替换为你的实例文件路径
    # 加载所有数据和参数
    prod_data = load_production_data(file_path)
    trans_data = load_transportation_data(file_path)
    drl_params = load_drl_hyperparameters()
    search_params = load_search_parameters()


    # 在你 __main__ 测试或 main.py 调用 env 前插入
    prod = prod_data  # 你已有的
    print("=== Production summary ===")
    total_ops = 0
    min_proc = float('inf')
    max_proc = 0
    sum_min_proc = 0
    for jid, ops in prod['jobs'].items():
        total_ops += len(ops)
        for op_id, m_times in ops:
            if m_times:
                mm = min(m_times.values())
                min_proc = min(min_proc, mm)
                max_proc = max(max_proc, mm)
                sum_min_proc += mm
    print(f"jobs: {len(prod['jobs'])}, total ops: {total_ops}")
    print(f"min op time: {min_proc}, max op time: {max_proc}, sum of min times (LB): {sum_min_proc}")


    # 1. 定义 T_internal 的搜索空间
    lower_bound, upper_bound = define_search_space(prod_data, trans_data)
    print(f"T_internal 搜索空间定义为: [{lower_bound}, {upper_bound}]")

    # 2. 顺序粗略网格搜索 (为解决挂起问题而修改)
    print("\n--- 开始顺序粗略网格搜索 (已禁用并行化) ---")
    coarse_search_space = np.arange(lower_bound, upper_bound, search_params['coarse_step'])
    
    results = []
    for t_internal in coarse_search_space:
        # 直接调用评估函数
        result = evaluate_t_internal(
            t_internal,
            prod_data=prod_data,
            trans_data=trans_data,
            drl_params=drl_params,
            max_episodes=drl_params['coarse_search_max_episodes']
        )
        results.append(result)

    # 3. 从结果中找到最佳解决方案
    best_total_cost = float('inf')
    best_t_internal = -1
    best_plan = None

    for t_internal, total_cost, plan, c_max in results:
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_t_internal = t_internal
            best_plan = plan
    
    print(f"\n--- 粗略搜索完成 ---")
    print(f"最佳 T_internal (粗略): {best_t_internal}, 成本: {best_total_cost}")


    # 4. 在最佳 T_internal 周围进行精细网格搜索
    if best_t_internal != -1:
        print("\n--- 开始精细网格搜索 ---")
        fine_search_lower = max(lower_bound, best_t_internal - search_params['fine_search_range'] // 2)
        fine_search_upper = min(upper_bound, best_t_internal + search_params['fine_search_range'] // 2)
        fine_search_space = np.arange(fine_search_lower, fine_search_upper, search_params['fine_step'])
        
        fine_results = []
        for t_internal in fine_search_space:
            # 避免重复计算
            if t_internal in [res[0] for res in results]:
                continue
            result = evaluate_t_internal(
                t_internal,
                prod_data=prod_data,
                trans_data=trans_data,
                drl_params=drl_params,
                max_episodes=drl_params['max_episodes'] # 使用完整的轮次进行精细搜索
            )
            fine_results.append(result)

        # 将精细搜索结果与之前的最佳结果合并
        results.extend(fine_results)
        results.sort(key=lambda x: x[0]) # 按 T_internal 排序以便绘图

        # 从所有结果中重新找到最佳解决方案
        for t_internal, total_cost, plan, c_max in results:
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_t_internal = t_internal
                best_plan = plan
                print(f"*** 找到新的最佳解决方案! T_internal={best_t_internal}, 成本={best_total_cost} ***")

    # 5. 可视化结果
    if results:
        plot_results(results)

    # --- 最终结果 ---
    print("\n\n--- HD-DRL 算法完成 ---")
    if best_t_internal != -1:
        print(f"最优 T_internal: {best_t_internal}")
        print(f"最低总成本: {best_total_cost}")
        if best_plan and best_plan.get('production'):
            c_max_final = max(op['end'] for op in best_plan['production'])
            print(f"最优生产计划 (C_max): {c_max_final}")
        else:
            print("未生成有效的生产计划。")
        
        if best_plan and best_plan.get('transport'):
            print("最优运输计划:")
            for market, plan in best_plan['transport'].items():
                print(f"  - {market}: {plan}")
        else:
            print("未生成有效的运输计划。")
    else:
        print("在给定的搜索空间和参数中未找到可行的解决方案。")

if __name__ == '__main__':
    # 注意: 由于已切换到顺序执行，多处理启动方法不再需要。
    # if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    #     multiprocessing.set_start_method('spawn', force=True)
    main()
