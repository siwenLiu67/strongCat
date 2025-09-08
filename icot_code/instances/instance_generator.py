#!/usr/bin/env python3
"""
generate_instances.py

按用户给定的参数范围自动生成 FJSP+multi-modal-transportation 实例（JSON）。
每个订单 = 最小生产单位（每个 job 对应一个 order_id）。

默认参数见脚本顶部（可通过 CLI 覆盖）。
输出目录: ./instances/
"""

import os
import json
import argparse
import random
from math import ceil
from datetime import datetime

# -----------------------
# 默认参数（按你给的表）
# -----------------------
DEFAULT_MACHINES_SET = [10, 20, 30]
DEFAULT_JOBS_SET = [20, 60, 100, 140]
OPS_PER_JOB_RANGE = (1, 20)              # n_i ~ U[1,20]
PROC_TIME_RANGE = (10, 50)               # t_ij ~ U[10,50]
UNIT_PRODUCTION_COST = 2.5               # C_(unit_production)

TRANSPORT_TYPES = ['Air', 'Sea', 'Land']
# 速度范围 (units: km/h)
SPEED_RANGES = {
    'Air': (800, 1000),
    'Land': (60, 100),
    'Sea': (25, 40)
}
# unit transportation cost mapping per type (you specified {1.2,0.5,0.2})
UNIT_TRANSPORT_COST = {
    'Air': 1.2,
    'Land': 0.5,
    'Sea': 0.2
}

# markets (example set; 你可以修改或传入自定义列表)
DEFAULT_MARKETS = ['US', 'Australia', 'Germany', 'Japan', 'France', 'Netherlands', 'Canada', 'UK']

# provider count per market per mode
PROVIDERS_PER_MODE = 3

# 距离范围 (km) 随机化（可据实际场景调整）
DISTANCE_RANGE = (2000, 10000)

ORDER_WEIGHT_RANGE = (1, 10)  # 订单重量范围 (tons)

# 订单截止因子范围：due = lower_bound_proc_time * factor, factor ~ U(1.2, 3.0)
DUE_FACTOR_RANGE = (2.0, 4.0)

# 每组参数生成的实例数（默认每 (machines, jobs) 生成 N inst）
DEFAULT_INSTANCES_PER_CFG = 3

# 输出目录
OUT_DIR = "instances"

# -----------------------
# 辅助函数
# -----------------------
def sample_speed(mode, rnd):
    lo, hi = SPEED_RANGES[mode]
    return rnd.uniform(lo, hi)

def sample_distance(rnd):
    return rnd.uniform(*DISTANCE_RANGE)

def make_transport_providers(rnd, markets, modes=TRANSPORT_TYPES, providers_per_mode=PROVIDERS_PER_MODE):
    """
    生成 transport_data 结构:
    { market: { mode: { provider_id: {cost_per_unit, speed, distance, capacity, lead_time_hours} } } }
    """
    transport = {}
    for market in markets:
        transport[market] = {}
        for mode in modes:
            transport[market][mode] = {}
            for p in range(1, providers_per_mode + 1):
                pid = f"P{p}"
                speed = sample_speed(mode, rnd)
                distance = sample_distance(rnd)
                # 简单估计 lead_time(hours) = distance / speed + some handling hours
                lead_time_hours = distance / max(speed, 1e-6) + rnd.uniform(24, 168)  # 加处理时间 1~7天
                # capacity: 以运输模式不同设置不同典型值（可调整）
                if mode == 'Air':
                    capacity = rnd.randint(50, 500)
                elif mode == 'Land':
                    capacity = rnd.randint(500, 5000)
                else:  # Sea
                    capacity = rnd.randint(1000, 20000)
                transport[market][mode][pid] = {
                    'cost_per_unit': UNIT_TRANSPORT_COST[mode],
                    'speed': round(speed, 2),
                    'distance': round(distance, 2),
                    'lead_time_hours': round(lead_time_hours, 2),
                    'capacity': capacity
                }
    return transport


import random
from math import ceil

def generate_jobs_and_orders(num_machines, num_jobs, rnd, markets, transport_data,
                             ops_range, proc_time_range, DUE_FACTOR_RANGE):
    """
    生成 JOBS 与 ORDERS：
    - 每个 job 直接对应一个 order (order_id)
    - job 格式: {job_id: {'ops': [(op_id, {machine: time, ...}), ...], 'order_id': OID, 'units':1}}
    - orders 格式: {order_id: {'market': market, 'qty':1, 'due': due, 'priority':..., 'can_split': False}}
    """
    machines = [f"M{i+1}" for i in range(num_machines)]
    jobs = {}
    orders = {}
    
    # 首先，计算所有作业的理论生产下限
    total_min_proc_time = 0
    max_min_job_time = 0
    job_details = {} # 用于临时存储每个作业的理想加工时间
    
    # 第一次循环：生成 jobs 和 ops，并计算每个作业的理想加工时间
    for j in range(1, num_jobs + 1):
        job_id = f"J{j}"
        order_id = f"O{j}"
        
        n_ops = rnd.randint(ops_range[0], ops_range[1])
        ops = []
        job_min_proc_time = 0
        
        for oi in range(1, n_ops + 1):
            op_name = f"O{j}_{oi}"
            m_count = rnd.randint(1, min(4, num_machines))
            m_candidates = rnd.sample(machines, m_count)
            m_times = {m: rnd.randint(proc_time_range[0], proc_time_range[1]) for m in m_candidates}
            ops.append((op_name, m_times))
            job_min_proc_time += min(m_times.values())
        
        jobs[job_id] = {'ops': ops, 'order_id': order_id, 'units': 1}
        job_details[job_id] = {'min_proc_time': job_min_proc_time}
        
        total_min_proc_time += job_min_proc_time
        if job_min_proc_time > max_min_job_time:
            max_min_job_time = job_min_proc_time

    # 计算全局生产下限
    # 使用总工时除以机器数，以及最长作业时间，取两者最大值
    if num_machines > 0:
        lb_production_time = max(ceil(total_min_proc_time / num_machines), max_min_job_time)
    else:
        lb_production_time = max_min_job_time
    
    # 第二次循环：生成 orders，使用全局生产下限计算截止时间
    for j in range(1, num_jobs + 1):
        job_id = f"J{j}"
        order_id = f"O{j}"
        
        market = rnd.choice(markets)
        
        min_transport_time_market = float('inf')
        if market in transport_data:
            for mode, providers in transport_data[market].items():
                for provider, params in providers.items():
                    transport_time = params['distance'] / max(params['speed'], 1e-6)
                    if transport_time < min_transport_time_market:
                        min_transport_time_market = transport_time
        
        if min_transport_time_market == float('inf'):
            print(f"Warning: No transport data for market {market}. Skipping due time calculation for this order.")
            continue
            
        # 核心修改：使用全局生产下限 lb_production_time 来计算 due
        lb_total_time = lb_production_time + min_transport_time_market
        factor = rnd.uniform(*DUE_FACTOR_RANGE)
        due = ceil(lb_total_time * factor)
        
        orders[order_id] = {
            'market': market,
            'qty': 1,
            'due': int(due),
            'weight': rnd.randint(*ORDER_WEIGHT_RANGE),  # 随机订单重量
            'priority': rnd.randint(1, 3),
            'can_split': False
        }
    
    # 确保 jobs 中的 order_id 引用在 orders 中存在
    # 如果因为 continue 导致 orders 列表有空缺，则需要处理
    # 这里的逻辑假设每个 job 都会有对应的 order
    
    return machines, jobs, orders


# def generate_jobs_and_orders(num_machines, num_jobs, rnd, markets,
#                              ops_range=OPS_PER_JOB_RANGE,
#                              proc_time_range=PROC_TIME_RANGE):
#     """
#     生成 JOBS 与 ORDERS：
#     - 每个 job 直接对应一个 order (order_id)
#     - job 格式: {job_id: {'ops': [(op_id, {machine: time, ...}), ...], 'order_id': OID, 'units':1}}
#     - orders 格式: {order_id: {'market': market, 'qty':1, 'due': due, 'priority':..., 'can_split': False}}
#     """
#     machines = [f"M{i+1}" for i in range(num_machines)]
#     jobs = {}
#     orders = {}

#     for j in range(1, num_jobs + 1):
#         job_id = f"J{j}"
#         order_id = f"O{j}"
#         # op_count ~ uniform
#         n_ops = rnd.randint(ops_range[0], ops_range[1])
#         ops = []
#         for oi in range(1, n_ops + 1):
#             op_name = f"O{j}_{oi}"
#             # 随机选择可用机器子集 (至少 1 台)
#             m_count = rnd.randint(1, min(4, num_machines))
#             m_candidates = rnd.sample(machines, m_count)
#             # 对每台机器指定处理时间 t_ij ~ U[proc_time_range]
#             m_times = {m: rnd.randint(proc_time_range[0], proc_time_range[1]) for m in m_candidates}
#             ops.append((op_name, m_times))
#         jobs[job_id] = {'ops': ops, 'order_id': order_id, 'units': 1}

#         # 对每个 job 计算下界加工时间（所有 ops 在其最快机器上之和）
#         lb_proc = sum(min(mtimes.values()) for _, mtimes in ops)
#         # due = lb_proc * factor
#         factor = rnd.uniform(*DUE_FACTOR_RANGE)
#         due = ceil(lb_proc * factor)
#         # pick market randomly
#         market = rnd.choice(markets)
#         orders[order_id] = {
#             'market': market,
#             'qty': 1,
#             'due': int(due),
#             'priority': rnd.randint(1, 3),
#             'can_split': False  # 遵循“订单为最小单位”的约束
#         }

#     return machines, jobs, orders

def make_instance_cfg(num_machines, num_jobs, rnd_seed, markets):
    rnd = random.Random(rnd_seed)
    
    transport_data = make_transport_providers(rnd, markets)
    machines, jobs, orders = generate_jobs_and_orders(
        num_machines, 
        num_jobs, 
        rnd, 
        markets, 
        transport_data,
        OPS_PER_JOB_RANGE,  # 传入 ops_range
        PROC_TIME_RANGE,    # 传入 proc_time_range
        DUE_FACTOR_RANGE    # 传入 DUE_FACTOR_RANGE
    )
    instance = {
        'meta': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'seed': rnd_seed,
            'num_machines': num_machines,
            'num_jobs': num_jobs,
            'unit_production_cost': UNIT_PRODUCTION_COST,
            'transport_types': TRANSPORT_TYPES
        },
        'machines': machines,          # list of machine ids
        'jobs': jobs,                  # job dict
        'orders': orders,              # order dict
        'transport_data': transport_data
    }
    return instance

# -----------------------
# CLI & 主流程
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate FJSP + transportation instances.")
    parser.add_argument("--machines", nargs="+", type=int, default=DEFAULT_MACHINES_SET,
                        help="List of machine counts to generate (e.g. 10 20 30)")
    parser.add_argument("--jobs", nargs="+", type=int, default=DEFAULT_JOBS_SET,
                        help="List of job counts to generate (e.g. 20 60 100)")
    parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS,
                        help="Market list")
    parser.add_argument("--per_cfg", type=int, default=DEFAULT_INSTANCES_PER_CFG,
                        help="Instances per (machines, jobs) config")
    parser.add_argument("--outdir", default=OUT_DIR, help="Output directory")
    parser.add_argument("--seed0", type=int, default=0, help="Base random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_seed = int(args.seed0)
    total = 0
    for m in args.machines:
        for j in args.jobs:
            for k in range(args.per_cfg):
                seed = base_seed + total + 1
                inst = make_instance_cfg(m, j, seed, args.markets)
                fname = f"instance_m{m}_j{j}_s{seed}.json"
                path = os.path.join(args.outdir, fname)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(inst, f, indent=2)
                print(f"Saved {path}")
                total += 1
    print(f"\nDone. Generated {total} instances in {args.outdir}")

if __name__ == "__main__":
    main()
