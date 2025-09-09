#!/usr/bin/env python3
"""
测试集成生产和运输环境的DQN算法
使用实际实例数据验证可行性检查功能
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icot_code.comparison_algorithms.single_layer_drl import DQNAlgorithm
from icot_code.data_loader import load_production_data, load_transportation_data
from icot_code.models.integrated_production_env import IntegratedProductionTransportEnv

def test_dqn_with_integrated_env():
    """测试使用集成环境的DQN算法"""
    print("=== 测试集成生产和运输环境的DQN算法 ===")
    
    # 使用实际实例数据
    instance_id = "instance_m10_j20_s1"
    instance_path = f'/Users/siwenliu/Desktop/my_project/strongCat/icot_code/instances/{instance_id}.json'
    
    if not os.path.exists(instance_path):
        print(f"错误: 实例文件 {instance_path} 不存在")
        return
    
    try:
        # 加载数据
        print(f"加载实例数据: {instance_id}")
        production_data = load_production_data(instance_path)
        transportation_data = load_transportation_data(instance_path)
        orders_data = transportation_data['orders']
        
        # 打印数据摘要
        print(f"作业数量: {len(production_data['jobs'])}")
        print(f"订单数量: {len(orders_data)}")
        print(f"市场数量: {len(transportation_data['transport_data'])}")
        
        # 测试集成环境
        print("\n=== 测试集成环境 ===")
        # 选择一个合适的 T_internal 值（基于订单交货期的中间值）
        T_internal = 1500
        env = IntegratedProductionTransportEnv(
            production_data, 
            orders_data, 
            T_internal, 
            transportation_data['transport_data']
        )
        print(f"状态空间维度: {env.observation_space.shape}")
        print(f"动作空间大小: {env.action_space.n}")
        
        # 重置环境
        state = env.reset()
        print(f"初始状态维度: {len(state)}")
        
        # 运行DQN算法
        print("\n=== 运行DQN算法 ===")
        dqn_algorithm = DQNAlgorithm()
        result = dqn_algorithm.solve(production_data, transportation_data, orders_data)
        
        # 打印结果
        print(f"算法名称: {result['algorithm']}")
        print(f"总成本: {result['metrics']['total_cost']}")
        print(f"生产完成时间: {result['metrics'].get('makespan', 'N/A')}")
        print(f"计算时间: {result['metrics']['computation_time']:.2f}秒")
        print(f"运输是否可行: {result['metrics']['transport_feasible']}")
        
        # 显示生产完成时间（C_max）和订单交货期对比
        if 'schedule' in result and result['schedule']:
            makespan = max([op['end'] for op in result['schedule']])
            print(f"实际生产完成时间: {makespan}")
            
            # 检查哪些订单可能超期
            print("\n订单交货期检查:")
            for job_id, job_info in production_data['jobs'].items():                        
                order_id = job_info[0].items()[0]  # 假设订单ID与第一个操作相关联
                if order_id in orders_data:
                    due_date = orders_data[order_id]['due']
                    if makespan > due_date:
                        print(f"  {order_id} ({orders_data[order_id]['market']}): 交货期={due_date}, 预计超期={makespan-due_date:.1f}")
                    else:
                        print(f"  {order_id} ({orders_data[order_id]['market']}): 交货期={due_date}, 剩余时间={due_date-makespan:.1f}")
        
        # 检查运输计划
        if 'transport_plan' in result and result['transport_plan']:
            print("\n运输计划详情:")
            for market, plan in result['transport_plan'].items():
                print(f"  {market}: {plan}")
        
        # 检查生产计划
        if 'production_schedule' in result and result['production_schedule']:
            print(f"\n生产计划包含 {len(result['production_schedule'])} 个操作")
            
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dqn_with_integrated_env()
