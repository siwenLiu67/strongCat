import numpy as np
import pandas as pd
from entity.dynamic_fjsp_env import WarehouseEnvironment
from data.caseBuilder.config import Config
from data.caseBuilder.jobshop_case_generator import FlexibleJobShopScenario

class EnvironmentTester:
    """环境测试器类"""
    
    def __init__(self):
        # 创建小规模测试用例
        self.config = self._create_test_config()
        self.case = FlexibleJobShopScenario(self.config)
        self.env = WarehouseEnvironment(self.config, self.case)
        
    def _create_test_config(self):
        """创建测试配置"""

        # 使用小规模问题进行测试
        config = Config()
        Config.problem.num_jobs = 5
        config.problem.num_machines = 3
        config.problem.num_distributors = 2
        config.problem.max_jobs_per_batch = 3
        config.problem.max_operations = 4
        config.max_time_steps = 10
        return config
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始环境测试...\n")
        
        # 测试初始化
        self.test_initialization()
        
        # 测试状态表示
        self.test_state_representation()
        
        # 测试动作执行
        self.test_action_execution()
        
        # 测试约束检查
        self.test_constraints()
        
        # 测试奖励计算
        self.test_rewards()
        
        print("\n所有测试完成!")
        
    def test_initialization(self):
        """测试环境初始化"""
        print("测试环境初始化...")
        
        # 重置环境
        state = self.env.reset()
        
        # 验证初始状态
        assert len(self.env.jobs) == self.config.problem.num_jobs, "作业数量不匹配"
        assert len(self.env.machines) == self.config.problem.num_machines, "机器数量不匹配"
        assert self.env.current_time == 0, "初始时间不为0"
        
        print("✓ 环境初始化测试通过")
        
    def test_state_representation(self):
        """测试状态表示"""
        print("\n测试状态表示...")
        
        state = self.env._get_state()
        
        # 验证元状态
        assert isinstance(state['meta'], np.ndarray), "元状态格式错误"
        assert len(state['meta']) == 1, "元状态维度错误"
        
        # 验证仓储状态
        warehouse = state['warehouse']
        assert warehouse['nodes']['jobs'].shape[0] == self.config.problem.num_jobs, "作业节点数量错误"
        assert warehouse['nodes']['machines'].shape[0] == self.config.problem.num_machines, "机器节点数量错误"
        
        print("✓ 状态表示测试通过")
        
    def test_action_execution(self):
        """测试动作执行"""
        print("\n测试动作执行...")
        
        # 创建测试动作
        actions = {
            'machine_assignment': [0 if i == 0 else None for i in range(self.config.problem.num_machines)],
            'batch_assignment': {0: '1_1260'}
        }
        
        # 执行动作
        next_state, reward, done, info = self.env.step(actions)
        
        # 验证执行结果
        assert isinstance(next_state, dict), "状态格式错误"
        assert isinstance(reward, (int, float)), "奖励格式错误"
        assert isinstance(done, bool), "终止标志格式错误"
        
        print("✓ 动作执行测试通过")
        
    def test_constraints(self):
        """测试约束检查"""
        print("\n测试约束检查...")
        
        # 测试机器分配约束
        job = self.env.jobs[0]
        machine = self.env.machines[0]
        valid = self.env._is_valid_assignment(job, machine)
        assert isinstance(valid, bool), "约束检查返回值类型错误"
        
        # 测试批次分配约束
        job = self.env.jobs[0]
        batch = self.env.batches[0]
        valid = self.env._is_valid_batch_assignment(job, batch)
        assert isinstance(valid, bool), "批次约束检查返回值类型错误"
        
        print("✓ 约束检查测试通过")
        
    def test_rewards(self):
        """测试奖励计算"""
        print("\n测试奖励计算...")
        
        # 测试调度奖励
        schedule_rewards = self.env._process_scheduling([None] * self.config.problem.num_machines)
        assert isinstance(schedule_rewards, (int, float)), "调度奖励格式错误"
        
        # 测试批次奖励
        batch_rewards = self.env._process_batching({})
        assert isinstance(batch_rewards, (int, float)), "批次奖励格式错误"
        
        # 测试总奖励
        total_reward = self.env._calculate_reward(schedule_rewards, batch_rewards)
        assert isinstance(total_reward, (int, float)), "总奖励格式错误"
        
        print("✓ 奖励计算测试通过")

def main():
    """主函数"""
    tester = EnvironmentTester()
    tester.run_all_tests()
    
if __name__ == "__main__":
    main()