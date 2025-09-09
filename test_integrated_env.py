#!/usr/bin/env python3
"""
测试集成生产和运输环境的脚本
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from icot_code.models.integrated_production_env import IntegratedProductionTransportEnv

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
    "market1": {"due": 100, "quantity": 1},
    "market2": {"due": 150, "quantity": 1}
}

transport_data = {
    "vehicles": [
        {"capacity": 10, "cost_per_km": 1.0, "speed": 50}
    ],
    "distances": {
        "factory": {"market1": 50, "market2": 80}
    }
}

# 测试环境创建
try:
    env = IntegratedProductionTransportEnv(
        production_data, orders_data, T_internal=200, transport_data=transport_data
    )
    print("✓ 环境创建成功")
    
    # 测试状态空间
    state = env.reset()
    print(f"✓ 状态空间大小: {len(state)}")
    print(f"✓ 动作空间大小: {env.action_space.n}")
    
    # 测试一步操作
    action = 0  # 选择第一个调度规则
    next_state, reward, done, info = env.step(action)
    print(f"✓ 单步操作成功，奖励: {reward}, 完成: {done}")
    
    print("\n✓ 集成环境测试通过！")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
