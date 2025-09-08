#!/usr/bin/env python3
"""
测试启发式算法实现
验证三个启发式算法的基本功能
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icot_code.comparison_algorithms.heuristic_algorithms import SPTRule, EDDRule, CompositeRule


def create_test_data():
    """创建测试数据"""
    # 简单的测试数据，符合实例文件格式
    production_data = {
        'jobs': {
            'J1': {
                'ops': [
                    ['O1_1', {'M1': 5, 'M2': 7}],
                    ['O1_2', {'M1': 3, 'M3': 4}]
                ],
                'order_id': 'O1',
                'units': 1
            },
            'J2': {
                'ops': [
                    ['O2_1', {'M2': 6, 'M3': 8}],
                    ['O2_2', {'M1': 4, 'M2': 5}]
                ],
                'order_id': 'O2',
                'units': 1
            }
        },
        'machines': ['M1', 'M2', 'M3'],
        'precedence': {
            'J1': {'O1_2': ['O1_1']},
            'J2': {'O2_2': ['O2_1']}
        },
        'c_unit_production': 10
    }
    
    orders_data = {
        'O1': {
            'market': 'TestMarket',
            'qty': 1,
            'due': 50,
            'weight': 1,
            'priority': 1,
            'can_split': False
        },
        'O2': {
            'market': 'TestMarket',
            'qty': 1,
            'due': 60,
            'weight': 1,
            'priority': 1,
            'can_split': False
        }
    }
    
    return production_data, orders_data


def test_algorithm(algorithm, name, production_data, orders_data, T_internal):
    """测试单个算法"""
    print(f"\n--- 测试 {name} 算法 ---")
    
    try:
        result = algorithm.solve(production_data, orders_data, T_internal)
        print(f"  成功! 结果:")
        print(f"  C_max: {result['metrics']['makespan']}")
        print(f"  调度方案长度: {len(result['schedule'])}")
        print(f"  性能指标: {result['metrics']}")
        
        # 验证基本结果格式
        assert 'schedule' in result, "缺少 schedule"
        assert 'metrics' in result, "缺少 metrics"
        assert 'makespan' in result['metrics'], "metrics 中缺少 makespan"
        assert isinstance(result['metrics']['makespan'], (int, float)), "makespan 应该是数字"
        assert isinstance(result['schedule'], list), "schedule 应该是列表"
        assert isinstance(result['metrics'], dict), "metrics 应该是字典"
        
        return True
        
    except Exception as e:
        import traceback
        print(f"  失败! 错误: {e}")
        print(f"  详细错误信息:")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=== 测试启发式算法实现 ===")
    
    # 创建测试数据
    production_data, orders_data = create_test_data()
    T_internal = 15
    
    # 初始化算法
    algorithms = [
        (SPTRule(), "SPT"),
        (EDDRule(), "EDD"), 
        (CompositeRule(), "Composite")
    ]
    
    success_count = 0
    total_tests = len(algorithms)
    
    for algorithm, name in algorithms:
        if test_algorithm(algorithm, name, production_data, orders_data, T_internal):
            success_count += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过测试: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("所有启发式算法实现正确!")
        return True
    else:
        print("部分算法测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
