#!/usr/bin/env python3
"""
测试DQN算法与集成环境的集成
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from icot_code.comparison_algorithms.single_layer_drl.dqn_algorithm import DQNAlgorithm

# 创建简单的测试数据
production_data = {
    "jobs": {
        "job1": [("op1", {"machine1": 10, "machine2": 15}), ("op2", {"machine1": 20})],
        "job2": [("op1", {"machine2": 12}), ("op2", {"machine1": 18})]
    },
    "machines": {
        "machine1": {"capacity": 1},
        "machine2": {"capacity": 1}
    },
    "precedence": {
        "job1": {"op2": ["op1"]},
        "job2": {"op2": ["op1"]}
    },
    "c_unit_production": 1.0
}

orders_data = {
    "O1": {"market": "market1", "due": 100, "qty": 1, "weight": 5, "priority": 1, "can_split": False},
    "O2": {"market": "market2", "due": 150, "qty": 1, "weight": 3, "priority": 2, "can_split": False}
}

transport_data = {
    "market1": {
        "Air": {
            "P1": {"cost_per_unit": 1.2, "speed": 800, "distance": 50, "capacity": 100}
        },
        "Sea": {
            "P1": {"cost_per_unit": 0.2, "speed": 30, "distance": 50, "capacity": 1000}
        }
    },
    "market2": {
        "Air": {
            "P1": {"cost_per_unit": 1.2, "speed": 800, "distance": 80, "capacity": 100}
        },
        "Sea": {
            "P1": {"cost_per_unit": 0.2, "speed": 30, "distance": 80, "capacity": 1000}
        }
    }
}

# 测试DQN算法
try:
    dqn = DQNAlgorithm()
    result = dqn.solve(production_data, transport_data, orders_data, num_episodes=2)
    
    print("✓ DQN算法运行成功")
    print(f"总成本: {result['metrics']['total_cost']}")
    print(f"生产可行: {result['metrics']['transport_feasible']}")
    print(f"计算时间: {result['metrics']['computation_time']:.2f}秒")
    
    print("\n✓ DQN与集成环境集成测试通过！")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
