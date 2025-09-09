# models/transportation_model.py

import math


def plan_transportation(C_max, transport_data, orders):
    """
    在给定生产完成时间 (C_max) 的情况下，为所有市场计算最低的运输成本。

    Args:
        C_max (float): 生产阶段的完成时间。
        transport_data (dict): 包含运输参数的字典。
        orders (dict): 包含每个市场订单详情的字典。

    Returns:
        tuple: 一个元组，包含:
            - total_transport_cost (float): 最低的总运输成本。
            - transport_plan (dict): 每个市场的详细运输计划。
                                     如果某个市场找不到可行的计划，则返回 None。
    """
    total_transport_cost = 0
    transport_plan = {}
    all_feasible = True

    
    for order_id, order_details in orders.items():
        deadline = order_details['due']
        weight = order_details['weight']
        market = order_details['market']
        
        
        best_option = None
        min_cost = float('inf')

        market_transport_options = transport_data.get(market, {})
        
        for mode, providers in market_transport_options.items():
            # 应用地理限制
            if market in ['US', 'Japan', 'Australia'] and mode == 'Land':
                continue

            for provider, params in providers.items():
                distance = params['distance']
                speed = params['speed']
                cost_per_km_ton = params['cost_per_unit']

                transport_time = distance / speed
                total_time = C_max + transport_time

                if total_time <= deadline:
                    current_cost = distance * cost_per_km_ton * weight
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_option = {
                            'mode': mode,
                            'provider': provider,
                            'cost': current_cost,
                            'transport_time': transport_time,
                            'total_time': total_time
                        }
        
        if best_option:
            transport_plan[market] = best_option
            total_transport_cost += min_cost
        else:
            # 如果任何一个市场找不到可行的选项，则整个计划不可行
            all_feasible = False
            transport_plan[market] = None
            # 分配一个非常高的成本以表示不可行
            total_transport_cost = float('inf')
            break # 无需检查其他市场

    return total_transport_cost, transport_plan

if __name__ == '__main__':
    # 使用示例:
    import sys
    import os
    # 将项目根目录添加到 sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from icot_code.data_loader import load_transportation_data, load_production_data
    path = '/Users/siwenliu/Desktop/my_project/strongCat/icot_code/instances/instance_m10_j20_s1.json'
    trans_data = load_transportation_data(path)
    prod_data = load_production_data(path)
    
    # --- 测试用例 1: 可行的 C_max ---
    test_C_max_1 = 300 
    cost1, plan1 = plan_transportation(test_C_max_1, trans_data['transport_data'], trans_data['orders'])
    print(f"--- 使用 C_max = {test_C_max_1} 进行测试 ---")
    print(f"总运输成本: {cost1}")
    print("运输计划:")
    for market, details in plan1.items():
        print(f"  {market}: {details}")

    print("-" * 30)

    # --- 测试用例 2: 不可行的 C_max ---
    test_C_max_2 = 800
    cost2, plan2 = plan_transportation(test_C_max_2, trans_data['transport_data'], trans_data['orders'])
    print(f"--- 使用 C_max = {test_C_max_2} 进行测试 ---")
    print(f"总运输成本: {cost2}")
    print("运输计划:")
    for market, details in plan2.items():
        print(f"  {market}: {details}")
