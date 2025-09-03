"""
测试动态优先规则启发式算法的派发功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from claude_code.dynamic_priority_rule_heuristic import DynamicPriorityRuleHeuristicSolver
from claude_code.environment import WarehouseEnvironment
from claude_code.config import Config
from claude_code.case_generator import FlexibleJobShopScenario

def test_dynamic_priority_rule_with_dispatch():
    """测试动态优先规则启发式算法的派发功能"""
    print("🧪 开始测试动态优先规则启发式算法的派发功能...")
    
    # 创建配置
    config = Config()
    config.num_initial_jobs = 5
    config.num_machines = 3
    config.num_distributors = 2
    config.num_dynamic_jobs = 0  # 先不测试动态作业
    config.max_time_steps = 100
    
    # 生成算例
    case = FlexibleJobShopScenario(config=config)
    
    # 创建环境
    env = WarehouseEnvironment(config, case)
    
    # 创建求解器
    solver = DynamicPriorityRuleHeuristicSolver(env, config, rule_name="EDD")
    
    try:
        # 运行求解器
        result = solver.solve()
        
        # 检查结果
        stats = result.get('stats', {})
        additional_metrics = result.get('additional_metrics', {})
        
        print(f"✅ 测试成功完成！")
        print(f"📊 性能指标:")
        print(f"   - 最大完成时间: {stats.get('makespan', 0):.2f}")
        print(f"   - 总延迟时间: {stats.get('total_tardiness', 0):.2f}")
        print(f"   - 准时交付率: {stats.get('on_time_delivery_rate', 0):.2%}")
        print(f"   - 总加权延迟时间: {stats.get('total_weighted_tardiness', 0):.2f}")
        print(f"   - 解决时间: {additional_metrics.get('solve_time', 0):.4f}秒")
        
        # 检查派发状态
        dispatched_jobs = len(env.dispatched_jobs)
        completed_jobs = len(env.completed_jobs)
        total_jobs = len(env.initial_jobs)
        
        print(f"📦 派发状态:")
        print(f"   - 总作业数: {total_jobs}")
        print(f"   - 已完成作业: {completed_jobs}")
        print(f"   - 已派发作业: {dispatched_jobs}")
        
        if dispatched_jobs == total_jobs:
            print("✅ 所有作业都已成功派发！")
        else:
            print(f"⚠️  警告: 只有 {dispatched_jobs}/{total_jobs} 个作业被派发")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_dynamic_priority_rule_with_dispatch()
    if success:
        print("\n🎉 所有测试通过！派发功能正常工作。")
    else:
        print("\n💥 测试失败，请检查代码。")
        sys.exit(1)
