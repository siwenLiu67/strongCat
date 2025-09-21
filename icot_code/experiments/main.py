#!/usr/bin/env python3
"""
运行单层深度强化学习算法对比实验
在多个实例上测试单层DRL算法的性能，并与启发式算法进行对比
"""

import sys
import os
import json
import time
from typing import Dict, List
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from icot_code.comparison_algorithms.heuristic_algorithms import SPTRule, EDDRule, CompositeRule
from icot_code.comparison_algorithms.single_layer_drl import DQNAlgorithm, PPOAlgorithm, A2CAlgorithm
from icot_code.data_loader import load_production_data, load_transportation_data, load_drl_hyperparameters, load_search_parameters
from icot_code.models.production_env import FJSPEnv
import icot_code.config
from icot_code.models.experiment_result import ExperimentResult, save_experiment_result
from agents.ppo_agent import PPOAgent
from agents.ppo_agent import DRL_Scheduling_Agent
from icot_code.models.transportation_model import plan_transportation
import numpy as np


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
    s_prod, c_max, rewards = DRL_Scheduling_Agent(env, agent, max_episodes)
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
    c_transport, s_trans = plan_transportation(c_max_for_transport, production_data=prod_data, orders=trans_data['orders'], transport_data=trans_data)

    print(f"  - 运输规划完成 (运输成本={c_transport}).")

    if c_transport == float('inf'):
        print(f"  - 对于生产完成时间={c_max_for_transport} 没有可行的运输计划。返回无限大成本。")
        return t_internal, float('inf'), None, None

    # c. 评估总成本
    c_production = c_max * prod_data['c_unit_production']
    total_cost = c_production + c_transport
    print(f"  - C_max: {c_max}, 生产成本:{c_production}, 运输成本: {c_transport}, 惩罚成本: {penalty_cost}, 总成本: {total_cost}")

    plan = {'production': s_prod, 'transport': s_trans, 'reward_curve': rewards}  # 新增reward_curve
    return t_internal, total_cost, plan, c_max


def main():
    """
    运行 HD-DRL 算法的主函数。
    """
    # 打印 PPO 代理将使用的设备
    from agents.ppo_agent import device
    print(f"DRL 代理将使用: {device}")

    instance_id = "instance_m10_j20_s1"  # 替换为你的实例ID
    file_path = '/Users/siwenliu/Desktop/my_project/strongCat/icot_code/instances/'+instance_id+'.json' # 替换为你的实例文件路径
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


    start_time = time.time()
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
    endtime = time.time()
    computation_time = endtime - start_time 

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


    # 进行结果的保存
    # 伪代码，适用于所有算法
    metrics = {
        "total_cost": best_total_cost,
        "production_cost": c_max_final * prod_data['c_unit_production'] if best_plan and best_plan.get('production') else float('inf'),
        "transportation_cost": best_total_cost - c_max_final * prod_data['c_unit_production'] ,
        "computation_time": computation_time,
    }
    result = ExperimentResult(
        algorithm_name="hrl_ppo",
        instance_id=instance_id,
        metrics=metrics,
        extra_info={}
    )
    save_experiment_result(result, save_dir="results", filetype="csv")


if __name__ == '__main__':
    # 注意: 由于已切换到顺序执行，多处理启动方法不再需要。
    # if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    #     multiprocessing.set_start_method('spawn', force=True)
    main()
