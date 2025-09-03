"""
测试脚本用于验证_complete_operation方法中的索引越界错误修复
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from case_generator import FlexibleJobShopScenario
from environment import WarehouseEnvironment
from data_structures import Job, Operation, Machine

def test_complete_operation_boundary_case():
    """测试_complete_operation方法在边界情况下的行为"""
    print("测试_complete_operation方法边界情况...")
    
    # 创建配置
    config = Config()
    config.num_initial_jobs = 1
    config.num_machines = 1
    config.num_distributors = 1
    config.min_operations = 1
    config.max_operations = 1
    
    # 创建测试场景
    scenario = FlexibleJobShopScenario(config=config)
    
    # 创建环境
    env = WarehouseEnvironment(config, scenario)
    
    # 获取第一个作业和机器
    job = env.available_jobs[0]
    machine = env.machines[0]
    
    print(f"初始状态:")
    print(f"  作业 {job.job_id}: 当前工序={job.current_operation}, 总工序数={len(job.operations)}")
    print(f"  机器 {machine.machine_id}: 状态={machine.status}")
    
    # 模拟机器开始加工作业
    machine.current_job = job.job_id
    machine.status = 'processing'
    machine.remaining_time = 5
    
    # 模拟时间推进，让机器完成加工
    env.t = 10  # 设置当前时间
    machine.remaining_time = 0  # 机器完成加工
    
    print(f"\n机器完成加工后:")
    print(f"  机器 {machine.machine_id}: 剩余时间={machine.remaining_time}")
    
    # 调用_complete_operation方法（这会触发边界条件）
    try:
        env._complete_operation(machine)
        print("✓ _complete_operation方法执行成功，没有抛出异常")
        
        # 检查作业状态
        print(f"\n_complete_operation执行后:")
        print(f"  作业 {job.job_id}: 状态={job.status}, 当前工序={job.current_operation}")
        print(f"  机器 {machine.machine_id}: 状态={machine.status}, 当前作业={machine.current_job}")
        
        # 验证作业是否被正确标记为完成
        if job.status == 'completed' and job in env.completed_jobs:
            print("✓ 作业正确标记为完成")
        else:
            print("✗ 作业未正确标记为完成")
            
        # 验证机器状态是否正确重置
        if machine.status == 'waiting' and machine.current_job == -1:
            print("✓ 机器状态正确重置")
        else:
            print("✗ 机器状态未正确重置")
            
    except IndexError as e:
        print(f"✗ _complete_operation方法抛出IndexError: {e}")
        return False
    except Exception as e:
        print(f"✗ _complete_operation方法抛出其他异常: {e}")
        return False
    
    return True

def test_multiple_operations_case():
    """测试多工序情况下的_complete_operation方法"""
    print("\n测试多工序情况...")
    
    # 创建配置
    config = Config()
    config.num_initial_jobs = 1
    config.num_machines = 1
    config.num_distributors = 1
    config.min_operations = 3
    config.max_operations = 3
    
    # 创建测试场景
    scenario = FlexibleJobShopScenario(config=config)
    
    # 创建环境
    env = WarehouseEnvironment(config, scenario)
    
    # 获取第一个作业和机器
    job = env.available_jobs[0]
    machine = env.machines[0]
    
    print(f"初始状态:")
    print(f"  作业 {job.job_id}: 当前工序={job.current_operation}, 总工序数={len(job.operations)}")
    
    # 模拟完成所有工序
    for op_index in range(len(job.operations)):
        # 模拟机器开始加工
        machine.current_job = job.job_id
        machine.status = 'processing'
        machine.remaining_time = 5
        
        # 模拟时间推进
        env.t += 10
        machine.remaining_time = 0
        
        print(f"\n完成工序 {op_index + 1}:")
        try:
            env._complete_operation(machine)
            print(f"  当前工序={job.current_operation}, 状态={job.status}")
            
            if op_index == len(job.operations) - 1:
                # 应该是最后一道工序
                if job.status == 'completed':
                    print("✓ 最后一道工序正确标记作业为完成")
                else:
                    print("✗ 最后一道工序未正确标记作业为完成")
            else:
                # 应该是中间工序
                if job.status == 'waiting':
                    print("✓ 中间工序正确标记作业为等待")
                else:
                    print("✗ 中间工序状态不正确")
                    
        except Exception as e:
            print(f"✗ 完成工序 {op_index + 1} 时出错: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("开始测试_complete_operation方法修复...")
    print("=" * 50)
    
    success1 = test_complete_operation_boundary_case()
    success2 = test_multiple_operations_case()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ 所有测试通过！索引越界错误修复成功。")
    else:
        print("✗ 部分测试失败，需要进一步检查修复。")
