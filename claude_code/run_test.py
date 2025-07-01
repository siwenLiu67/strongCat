from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement
from environment import WarehouseEnvironment
from high_level_agent import HighLevelAgent
from schedule_agent import ScheduleAgent
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
    # 预定义复杂测试动作序列
    test_actions = [
        # 第0步：分配两个作业的第一道工序
        {'schedule': {0: 0, 1: 1}},  # 作业0->机器0, 作业1->机器2
        {'wait': True},  # 等待加工
        {'wait': True},
        
        # 第3步：分配作业2的第一道工序
        {'schedule': {2: 1}},  # 作业2->机器1
        {'wait': True},
        {'wait': True},
        
        # 第6步：第一批作业完成后分配第二道工序
        {'schedule': {0: 1, 1: 0}},  # 作业0->机器2, 作业1->机器0
        {'wait': True},
        {'wait': True},
        
        # 第9步：创建第一个配送批次
        {'dispatch': {0: [0, 1]}},  # 批次0包含作业0,1
        {'wait': True},
        {'wait': True},
        
        # 第12步：分配作业2的最后工序
        {'schedule': {2: 1}},  # 作业2->机器2
        {'wait': True},
        {'wait': True},
        
        # 第15步：创建第二个配送批次
        {'dispatch': {1: [2]}},  # 批次1包含作业2
        {'wait': True},
        {'wait': True}
    ]

    meta_agent = HighLevelAgent(config)
    
    

    # 3. 执行循环
    # 测试environment
    env = WarehouseEnvironment(config, case)
    state = env.reset()
    done = False
    step = 0
    
    print("\n测试环境运行:")
    while not done:
        # 1. 高层动作决策
        meta_action, _ = meta_agent.select_action(state)
        # 0: 调度, 1: 配送, 2: 等待

        # 2. 低层动作（这里只用简单规则/随机，实际可接入调度/配送策略）
        if meta_action == 0:
            # 简单调度规则：将所有waiting作业分配到空闲机器
            action = {'schedule': {}}
            # 根据schedule_agent 进行动作选择
            scheduling_agent = ScheduleAgent(config)
            for job in state['available_jobs']:
                if hasattr(job, 'status') and job.status == 'waiting':
                    for machine in state['machines']:
                        if machine.status == 'waiting':
                            action['schedule'][job.job_id] = machine.machine_id
                            break
            if not action['schedule']:
                action = {'wait': True}
        elif meta_action == 1:
            # 简单配送规则：将所有已完成作业组成一个批次
            completed = [job.job_id for job in state['completed_jobs'] if hasattr(job, 'status') and job.status == 'completed']
            if completed:
                action = {'dispatch': {step: completed}}
            else:
                action = {'wait': True}
        else:
            action = {'wait': True}

        # 3. 环境交互
        next_state, reward, done, info = env.step(action)

        # 4. 打印结果
        print(f"\n时间步 {step}:")
        print(f"高层动作: {meta_action}，执行动作: {action}")
        print(f"- 可用作业数量: {len(next_state['available_jobs'])}")
        print(f"- 完成作业数量: {len(next_state['completed_jobs'])}")
        print(f"- 配送作业数量: {len(next_state.get('dispatched_jobs', []))}")
        print(f"- 奖励值: {reward:.2f}")

        state = next_state
        step += 1

        if done:
            print("\n环境结束!")
            break



if __name__ == "__main__":
    main()