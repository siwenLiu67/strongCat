from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
def main():
    """主函数"""
    # 1. 读取配置
    config = Config()

    # 2. 创建算例
    case = FlexibleJobShopScenario(config)
    # 打印算例信息
    print("生成的FJSP-DP算例:")
    print(f"作业数量: {len(case.jobs)}")
    print(f"机器数量: {len(case.machines)}")
    print(f"配送商数量: {len(case.distributors)}")


    # 预定义测试动作序列
    test_actions = [
        # 第0步：分配第一道工序到机器0
        {'schedule': {0: 0}},
         # 等待配送完成
        {'wait': True},
        # 第1步：分配第二道工序到机器1
        {'schedule': {0: 1}},
        
        # 第2步：创建配送批次
        {'dispatch': {0: [0]}},
         # 等待配送完成
        {'wait': True}
         ]
        
    # 3. 执行循环
    # 测试environment
    env = WarehouseEnvironment(config, case)
    state = env.reset()
    
    print("\n测试环境运行:")
    for step, action in enumerate(test_actions):
        
        # 执行环境步进
        next_state, reward, done, info = env.step(action)
        
        # 打印结果状态
        print("\n执行结果:")
        print(f"- 可用作业数量: {len(next_state['available_jobs'])}")
        print(f"- 完成作业数量: {len(next_state['completed_jobs'])}")
        print(f"- 配送作业数量: {len(next_state['dispatched_jobs'])}")
        print(f"- 奖励值: {reward:.2f}")
        
        if done:
            print("\n环境结束!")
            break
            
        state = next_state



if __name__ == "__main__":
    main()