# models/transportation_model.py

import math


def plan_transportation(C_max, production_data, orders, transport_data, penalty_per_time=1000):
    """
    在给定生产完成时间 (C_max) 的情况下，为所有市场计算最低的运输成本。
    如果不可行，则将超时部分按惩罚成本计入总成本。
    
    Args:
        C_max(float): 生产阶段的完成时间。
        transport_data (dict): 包含运输参数的字典。
        orders (dict): 包含每个市场订单详情的字典。
        penalty_per_time (float): 单位超时时间的惩罚成本。
    Returns:
        tuple: (总运输+惩罚成本, 运输计划)
    """
    total_transport_cost = 0
    transport_plan = {}
    
    for order_id, order_details in orders.items():
        deadline = order_details['due']
        weight = order_details['weight']
        market = order_details['market']

        best_option = None
        min_cost = float('inf')
        # 新增：记录最小超时方案
        min_overtime = float('inf')
        overtime_option = None
        overtime_cost = None

        
        market_transport_options = transport_data.get('transport_data', {}).get(market, {})

        for mode, providers in market_transport_options.items():
            if market in ['US', 'Japan', 'Australia'] and mode == 'Land':
                continue
            for provider, params in providers.items():
                distance = params['distance']
                speed = params['speed']
                cost_per_km_ton = params['cost_per_unit']
                transport_time = distance / speed
                total_time = C_max + transport_time 
                current_cost = distance * cost_per_km_ton * weight * total_time/1000
                if total_time <= deadline:
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_option = {
                            'mode': mode,
                            'provider': provider,
                            'cost': current_cost,
                            'transport_time': transport_time,
                            'total_time': total_time,
                            'overtime': 0,
                            'penalty': 0
                        }
                else:
                    overtime = total_time - deadline
                    if overtime < min_overtime:
                        min_overtime = overtime
                        overtime_cost = current_cost
                        overtime_option = {
                            'mode': mode,
                            'provider': provider,
                            'cost': current_cost,
                            'transport_time': transport_time,
                            'total_time': total_time,
                            'overtime': overtime,
                            'penalty': overtime * penalty_per_time
                        }
        if best_option:
            transport_plan[market] = best_option
            total_transport_cost += min_cost
        elif overtime_option:
            # 不可行，采用最小超时方案并加惩罚
            transport_plan[market] = overtime_option
            total_transport_cost += overtime_cost + overtime_option['penalty']
        else:
            # 没有任何方案
            transport_plan[market] = None
            total_transport_cost = float('inf')
            break
    return total_transport_cost, transport_plan

