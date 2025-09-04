"""
Integrated Flexible Job Shop Scheduling and Dispatching Problem (IFJSSP-DP) Model
Using OR-Tools CP-SAT Solver

Based on the mathematical formulation in Section 3 of the paper.
"""

from ortools.sat.python import cp_model
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

# 导入算例生成器和配置
from case_generator import FlexibleJobShopScenario
from config import Config


@dataclass
class MathModelDeliveryRequirement:
    """数学模型专用的交付需求数据结构"""
    requirement_id: int
    total_demand: int
    min_fulfillment_ratio: float
    penalty_weight: float
    job_contributions: Dict[int, int]  # job_id -> quantity


class IFJSSPDPModel:
    """
    集成柔性作业车间调度与派遣问题的OR-Tools实现
    """
    
    def __init__(self, 
                 scenario: FlexibleJobShopScenario,
                 max_batches: int = 100,
                 big_m: int = 10000):
        """
        初始化模型
        
        Args:
            scenario: 从case_generator生成的场景
            max_batches: 最大批次数量
            big_m: 大M常数
        """
        self.scenario = scenario
        self.jobs = scenario.jobs
        self.machines = scenario.machines
        self.distributors = scenario.distributors
        self.max_batches = max_batches
        self.big_m = big_m
        
        # 转换交付需求格式
        self.delivery_requirements = self._convert_delivery_requirements()
        
        # 创建机器ID列表
        self.machine_ids = [m.machine_id for m in self.machines]
        
        # 创建CP-SAT模型
        self.model = cp_model.CpModel()
        
        # 决策变量
        self.x = {}  # x[j,o,m]: 工序j.o是否分配给机器m
        self.s = {}  # s[j,o]: 工序j.o的开始时间
        self.c = {}  # c[j]: 作业j的完成时间
        self.t = {}  # t[j]: 作业j的延误时间
        self.y = {}  # y[j,b]: 作业j是否分配给批次b
        self.d = {}  # d[b]: 批次b的派遣时间
        self.z = {}  # z[b,r]: 批次b是否服务交付需求r
        self.w = {}  # w[j,b,r]: 作业j在批次b中是否贡献给需求r
        self.seq = {}  # seq[j1,o1,j2,o2,m]: 工序(j1,o1)是否在机器m上先于(j2,o2)
        self.u = {}  # u[r]: 交付需求r的短缺量
        
        # 辅助变量
        self.makespan_var = None
        
        self._create_variables()
        self._add_constraints()
    
    def _convert_delivery_requirements(self) -> List[MathModelDeliveryRequirement]:
        """将case_generator的交付需求转换为数学模型格式"""
        converted_requirements = []
        req_id = 0
        
        for distributor in self.distributors:
            if hasattr(distributor, 'delivery_requirements') and distributor.delivery_requirements:
                delivery_req = distributor.delivery_requirements
                
                # 获取分配给该配送商的作业
                assigned_jobs = [job for job in self.jobs if job.distributor_id == distributor.distributor_id]
                
                # 为每个时间窗口创建一个交付需求
                for i, (due_time, ratio, weight) in enumerate(zip(
                    delivery_req.due_times, 
                    delivery_req.ratios, 
                    delivery_req.weights
                )):
                    # 计算总需求量
                    total_demand = sum(job.amount for job in assigned_jobs)
                    
                    # 创建作业贡献字典
                    job_contributions = {job.job_id: job.amount for job in assigned_jobs}
                    
                    # 创建交付需求
                    math_req = MathModelDeliveryRequirement(
                        requirement_id=req_id,
                        total_demand=int(total_demand * ratio),
                        min_fulfillment_ratio=0.8,  # 默认最小履行比例
                        penalty_weight=weight,
                        job_contributions=job_contributions
                    )
                    converted_requirements.append(math_req)
                    req_id += 1
        
        return converted_requirements
        
    def _create_variables(self):
        """创建决策变量"""
        # 在 _create_variables 方法中添加
        self.is_batch_for_distributor = {}  # is_batch_for_distributor[b, r]: 批次b是否服务于配送商r
        for batch_id in range(self.max_batches):
            for distributor in self.distributors:
                var_name = f"is_batch_for_dist_{batch_id}_{distributor.distributor_id}"
                self.is_batch_for_distributor[batch_id, distributor.distributor_id] = self.model.NewBoolVar(var_name)

        self.delivery_time_var = {}
        for batch_id in range(self.max_batches):
            var_name = f"delivery_time_{batch_id}"
            self.delivery_time_var[batch_id] = self.model.NewIntVar(0, self.big_m * 2, var_name)
            
        # 新增变量: 作业的最终交付时间 job_delivery_time[j]
        self.job_delivery_time = {}
        for job in self.jobs:
            var_name = f"job_delivery_time_{job.job_id}"
            self.job_delivery_time[job.job_id] = self.model.NewIntVar(0, self.big_m * 3, var_name)


        # 工序分配变量 x[j,o,m]
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                # 使用operation的available_machine_ids属性
                eligible_machines = getattr(operation, 'available_machine_ids', 
                                          getattr(operation, 'eligible_machines', []))
                for machine_id in eligible_machines:
                    var_name = f"x_{job.job_id}_{op_idx}_{machine_id}"
                    self.x[job.job_id, op_idx, machine_id] = self.model.NewBoolVar(var_name)
        
        # 计算最大时间
        max_time = 10000  # 默认值
        if self.jobs:
            total_processing_time = 0
            for job in self.jobs:
                for operation in job.operations:
                    if hasattr(operation, 'processing_times') and operation.processing_times:
                        total_processing_time += min(operation.processing_times.values())
            max_time = total_processing_time + self.big_m*10
        
        # 工序开始时间 s[j,o]
        for job in self.jobs:
            for op_idx in range(len(job.operations)):
                var_name = f"s_{job.job_id}_{op_idx}"
                self.s[job.job_id, op_idx] = self.model.NewIntVar(0, max_time, var_name)
        
        # 作业完成时间 c[j]
        for job in self.jobs:
            var_name = f"c_{job.job_id}"
            self.c[job.job_id] = self.model.NewIntVar(0, max_time, var_name)
        
        # 作业延误时间 t[j]
        for job in self.jobs:
            var_name = f"t_{job.job_id}"
            self.t[job.job_id] = self.model.NewIntVar(0, max_time, var_name)
        
        # 作业批次分配 y[j,b]
        for job in self.jobs:
            for batch_id in range(self.max_batches):
                var_name = f"y_{job.job_id}_{batch_id}"
                self.y[job.job_id, batch_id] = self.model.NewBoolVar(var_name)
        
        # 批次派遣时间 d[b]
        for batch_id in range(self.max_batches):
            var_name = f"d_{batch_id}"
            self.d[batch_id] = self.model.NewIntVar(0, max_time, var_name)
        
        # 批次服务需求 z[b,r]
        for batch_id in range(self.max_batches):
            for req in self.delivery_requirements:
                var_name = f"z_{batch_id}_{req.requirement_id}"
                self.z[batch_id, req.requirement_id] = self.model.NewBoolVar(var_name)
        
        # 作业贡献变量 w[j,b,r]
        for job in self.jobs:
            for batch_id in range(self.max_batches):
                for req in self.delivery_requirements:
                    var_name = f"w_{job.job_id}_{batch_id}_{req.requirement_id}"
                    self.w[job.job_id, batch_id, req.requirement_id] = self.model.NewBoolVar(var_name)
        
        # 工序排序变量 seq[j1,o1,j2,o2,m]
        for machine_id in self.machine_ids:
            machine_operations = []
            for job in self.jobs:
                for op_idx, operation in enumerate(job.operations):
                    eligible_machines = getattr(operation, 'available_machine_ids', 
                                              getattr(operation, 'eligible_machines', []))
                    if machine_id in eligible_machines:
                        machine_operations.append((job.job_id, op_idx))
            
            for i, (j1, o1) in enumerate(machine_operations):
                for j, (j2, o2) in enumerate(machine_operations):
                    if i != j:
                        var_name = f"seq_{j1}_{o1}_{j2}_{o2}_{machine_id}"
                        self.seq[j1, o1, j2, o2, machine_id] = self.model.NewBoolVar(var_name)
        
        # 交付短缺变量 u[r]
        for req in self.delivery_requirements:
            var_name = f"u_{req.requirement_id}"
            self.u[req.requirement_id] = self.model.NewIntVar(0, req.total_demand, var_name)
        
        # Makespan变量（用于优化）
        self.makespan_var = self.model.NewIntVar(0, max_time, "makespan")
    
    def _add_constraints(self):
        """添加约束条件"""
        # 在 _add_constraints 方法中添加
        # 新增约束：批次中的所有作业必须属于同一个配送商
        for batch_id in range(self.max_batches):
            for job in self.jobs:
                # 如果作业 j 被分配到批次 b
                self.model.AddImplication(self.y[job.job_id, batch_id], 
                                        self.is_batch_for_distributor[batch_id, job.distributor_id])

        # 额外约束：一个批次最多服务一个配送商
        for batch_id in range(self.max_batches):
            self.model.Add(sum(self.is_batch_for_distributor[batch_id, d.distributor_id] for d in self.distributors) <= 1)
        # 约束8: 计算批次大小
        batch_size_var = {}
        for batch_id in range(self.max_batches):
            var_name = f"batch_size_{batch_id}"
            batch_size_var[batch_id] = self.model.NewIntVar(0, len(self.jobs), var_name)
            
            # 批次大小等于分配给该批次的作业数
            self.model.Add(
                batch_size_var[batch_id] == sum(self.y[job.job_id, batch_id] for job in self.jobs)
            )

        # 约束9: 计算批次配送完成时间
        # 定义常量
        BASE_DELIVERY_TIME = 10 
        PER_JOB_TIME = 5

        for batch_id in range(self.max_batches):
            # 批次完成配送的时间 = 派遣时间 + 配送时间
            self.model.Add(
                self.delivery_time_var[batch_id] == 
                self.d[batch_id] + BASE_DELIVERY_TIME + PER_JOB_TIME * batch_size_var[batch_id]
            )

        # 约束1: 工序分配约束 - 每个工序必须分配给一台机器
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                eligible_machines = getattr(operation, 'available_machine_ids', 
                                          getattr(operation, 'eligible_machines', []))
                if eligible_machines:
                    self.model.Add(
                        sum(self.x[job.job_id, op_idx, m] 
                            for m in eligible_machines) == 1
                    )
        
        # 约束2: 工序优先级约束 - 同一作业内工序的先后顺序
        for job in self.jobs:
            for op_idx in range(len(job.operations) - 1):
                operation = job.operations[op_idx]
                eligible_machines = getattr(operation, 'available_machine_ids', 
                                          getattr(operation, 'eligible_machines', []))
                processing_times = getattr(operation, 'processing_times', {})
                
                if eligible_machines and processing_times:
                    processing_time_expr = sum(
                        self.x[job.job_id, op_idx, m] * processing_times.get(m, 0)
                        for m in eligible_machines
                    )
                    self.model.Add(
                        self.s[job.job_id, op_idx + 1] >= 
                        self.s[job.job_id, op_idx] + processing_time_expr
                    )
        
        # 约束3: 到达时间约束
        for job in self.jobs:
            arrival_time = getattr(job, 'arrival_time', 0)
            
            # 核心修复：将浮点数转换为整数
            if isinstance(arrival_time, float):
                arrival_time = int(arrival_time)
            
            self.model.Add(self.s[job.job_id, 0] >= arrival_time)

        # 改进后的析取约束 (替换你原来的4 & 5)
        # 约束4 & 5: 同一机器上的工序不能重叠
        for machine_id in self.machine_ids:
            intervals = []
            # 遍历所有作业和工序，找出分配给当前机器的工序
            for job in self.jobs:
                for op_idx, operation in enumerate(job.operations):
                    eligible_machines = getattr(operation, 'available_machine_ids', 
                                                getattr(operation, 'eligible_machines', []))
                    
                    if machine_id in eligible_machines:
                        # 确保处理时间是整数
                        processing_time = int(operation.processing_times[machine_id])
                        
                        # 创建 IntervalVar，并只在工序分配给该机器时生效
                        interval_var = self.model.NewOptionalIntervalVar(
                            self.s[job.job_id, op_idx],
                            processing_time,
                            self.s[job.job_id, op_idx] + processing_time,
                            self.x[job.job_id, op_idx, machine_id],
                            f"interval_{job.job_id}_{op_idx}_{machine_id}"
                        )
                        intervals.append(interval_var)
            
            # 对当前机器上的所有工序添加无重叠约束
            self.model.AddNoOverlap(intervals)


        # 约束6: 作业完成时间
        for job in self.jobs:
            if job.operations:
                last_op_idx = len(job.operations) - 1
                last_operation = job.operations[last_op_idx]
                eligible_machines = getattr(last_operation, 'available_machine_ids', 
                                          getattr(last_operation, 'eligible_machines', []))
                processing_times = getattr(last_operation, 'processing_times', {})
                
                if eligible_machines and processing_times:
                    processing_time_expr = sum(
                        self.x[job.job_id, last_op_idx, m] * processing_times.get(m, 0)
                        for m in eligible_machines
                    )
                    self.model.Add(
                        self.c[job.job_id] >= 
                        self.s[job.job_id, last_op_idx] + processing_time_expr
                    )
        
        
        
        # 约束9: 每个作业分配给一个批次
        for job in self.jobs:
            self.model.Add(
                sum(self.y[job.job_id, b] for b in range(self.max_batches)) == 1
            )
        
        # 约束10: 批次派遣时间约束
        for job in self.jobs:
            for batch_id in range(self.max_batches):
                self.model.Add(
                    self.d[batch_id] >= self.c[job.job_id] - 
                    self.big_m * (1 - self.y[job.job_id, batch_id])
                )
        
        # 约束11-13: w变量的逻辑约束
        for job in self.jobs:
            for batch_id in range(self.max_batches):
                for req in self.delivery_requirements:
                    # w[j,b,r] <= y[j,b]
                    self.model.Add(
                        self.w[job.job_id, batch_id, req.requirement_id] <= 
                        self.y[job.job_id, batch_id]
                    )
                    # w[j,b,r] <= z[b,r]
                    self.model.Add(
                        self.w[job.job_id, batch_id, req.requirement_id] <= 
                        self.z[batch_id, req.requirement_id]
                    )
                    # w[j,b,r] >= y[j,b] + z[b,r] - 1
                    self.model.Add(
                        self.w[job.job_id, batch_id, req.requirement_id] >= 
                        self.y[job.job_id, batch_id] + 
                        self.z[batch_id, req.requirement_id] - 1
                    )
        
        # 约束14: 交付需求覆盖约束
        for req in self.delivery_requirements:
            if req.job_contributions:
                fulfilled_quantity = sum(
                    req.job_contributions.get(job.job_id, 0) * 
                    self.w[job.job_id, batch_id, req.requirement_id]
                    for job in self.jobs
                    for batch_id in range(self.max_batches)
                    if job.job_id in req.job_contributions
                )
                required_quantity = int(req.min_fulfillment_ratio * req.total_demand)
                self.model.Add(
                    fulfilled_quantity >= required_quantity - self.u[req.requirement_id]
                )
        
        # Makespan约束
        for job in self.jobs:
            self.model.Add(self.makespan_var >= self.c[job.job_id])


        # 定义一个辅助变量来表示作业所属的批次索引
        self.job_batch_index = {}
        for job in self.jobs:
            var_name = f"job_batch_index_{job.job_id}"
            self.job_batch_index[job.job_id] = self.model.NewIntVar(0, self.max_batches - 1, var_name)
            
            # 约束：job_batch_index 的值等于 y[job, batch_id] 为 1 的那个 batch_id
            self.model.Add(self.job_batch_index[job.job_id] == sum(
                b * self.y[job.job_id, b] for b in range(self.max_batches)
            ))


        # 修正后的作业最终交付时间计算（使用大M法）
        # 新增约束: 作业的最终交付时间
        self.job_delivery_time = {}
        for job in self.jobs:
            var_name = f"job_delivery_time_{job.job_id}"
            self.job_delivery_time[job.job_id] = self.model.NewIntVar(0, self.big_m * 3, var_name)
            
            # 建立作业交付时间与批次交付时间之间的关系
            for batch_id in range(self.max_batches):
                # 如果 y[job.job_id, batch_id] == 1，则 job_delivery_time[job.job_id] == delivery_time_var[batch_id]
                # 这个逻辑需要通过两个约束来实现
                
                # 约束1: T_j >= T_b - M * (1 - y_jb)
                self.model.Add(self.job_delivery_time[job.job_id] >= self.delivery_time_var[batch_id] - self.big_m * (1 - self.y[job.job_id, batch_id]))
                
                # 2. T_j <= T_b + M * (1 - y_jb)
                self.model.Add(self.job_delivery_time[job.job_id] <= self.delivery_time_var[batch_id] + self.big_m * (1 - self.y[job.job_id, batch_id]))

        # 约束7: 延误时间计算
        for job in self.jobs:
            due_date = getattr(job, 'due_date', 100)
            self.model.Add(self.t[job.job_id] >= self.job_delivery_time[job.job_id] - due_date)
            self.model.Add(self.t[job.job_id] >= 0)
            
    def set_objective(self, objective_type: str = "weighted"):
        """
        设置目标函数
        
        Args:
            objective_type: "tardiness" (最小化延误), "makespan" (最小化完工时间), 
                          "weighted" (加权目标)
        """
        
        # 加权目标：延误 + 交付短缺惩罚
        tardiness_term = sum(self.t[job.job_id] for job in self.jobs)
        shortage_penalty = sum(
                req.penalty_weight * self.u[req.requirement_id]
                for req in self.delivery_requirements
            )
        self.model.Minimize(tardiness_term + shortage_penalty)
        
       
    
    def solve(self, time_limit) -> Dict:
        """
        求解模型
        
        Args:
            time_limit: 求解时间限制（秒）
            
        Returns:
            包含求解结果的字典
        """
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = True
        
        start_time = time.time()
        status = solver.Solve(self.model)
        solve_time = time.time() - start_time
        
        result = {
            "status": solver.StatusName(status),
            "solve_time": solve_time,
            "objective_value": None,
            "schedule": {},
            "batches": {},
            "job_completion_times": {},
            "job_tardiness": {},
            "delivery_shortfalls": {},
            "scenario_info": {
                "num_jobs": len(self.jobs),
                "num_machines": len(self.machines),
                "num_distributors": len(self.distributors),
                "num_delivery_requirements": len(self.delivery_requirements)
            }
        }
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            result["objective_value"] = solver.ObjectiveValue()
            
            # 提取调度结果
            for job in self.jobs:
                job_schedule = []
                for op_idx, operation in enumerate(job.operations):
                    eligible_machines = getattr(operation, 'available_machine_ids', 
                                              getattr(operation, 'eligible_machines', []))
                    processing_times = getattr(operation, 'processing_times', {})
                    
                    for machine_id in eligible_machines:
                        if (job.job_id, op_idx, machine_id) in self.x:
                            if solver.Value(self.x[job.job_id, op_idx, machine_id]) == 1:
                                start_time = solver.Value(self.s[job.job_id, op_idx])
                                proc_time = processing_times.get(machine_id, 0)
                                job_schedule.append({
                                    "operation": op_idx,
                                    "machine": machine_id,
                                    "start_time": start_time,
                                    "processing_time": proc_time,
                                    "end_time": start_time + proc_time
                                })
                
                result["schedule"][job.job_id] = job_schedule
                result["job_completion_times"][job.job_id] = solver.Value(self.c[job.job_id])
                result["job_tardiness"][job.job_id] = solver.Value(self.t[job.job_id])
            
            # 提取批次结果
            for batch_id in range(self.max_batches):
                batch_jobs = []
                for job in self.jobs:
                    if solver.Value(self.y[job.job_id, batch_id]) == 1:
                        batch_jobs.append(job.job_id)
                
                if batch_jobs:  # 只记录非空批次
                    result["batches"][batch_id] = {
                        "jobs": batch_jobs,
                        "dispatch_time": solver.Value(self.d[batch_id]),
                        "served_requirements": []
                    }
                    
                    # 检查服务的交付需求
                    for req in self.delivery_requirements:
                        if solver.Value(self.z[batch_id, req.requirement_id]) == 1:
                            result["batches"][batch_id]["served_requirements"].append(req.requirement_id)
            
            # 提取交付短缺
            for req in self.delivery_requirements:
                result["delivery_shortfalls"][req.requirement_id] = solver.Value(self.u[req.requirement_id])
        
        return result
    
    def print_solution(self, result: Dict):
        """打印求解结果"""
        print(f"求解状态: {result['status']}")
        print(f"求解时间: {result['solve_time']:.2f}秒")
        
        # 打印场景信息
        if "scenario_info" in result:
            info = result["scenario_info"]
            print(f"\n=== 场景信息 ===")
            print(f"作业数量: {info['num_jobs']}")
            print(f"机器数量: {info['num_machines']}")
            print(f"配送商数量: {info['num_distributors']}")
            print(f"交付需求数量: {info['num_delivery_requirements']}")
        
        if result['objective_value'] is not None:
            print(f"目标函数值: {result['objective_value']}")
            
            print("\n=== 作业调度结果 ===")
            for job_id, schedule in result["schedule"].items():
                if schedule:  # 只显示有调度的作业
                    print(f"作业 {job_id}:")
                    for op in schedule:
                        print(f"  工序{op['operation']}: 机器{op['machine']}, "
                              f"时间[{op['start_time']}, {op['end_time']}]")
                    print(f"  完成时间: {result['job_completion_times'][job_id]}, "
                          f"延误: {result['job_tardiness'][job_id]}")
            
            print("\n=== 批次派遣结果 ===")
            for batch_id, batch_info in result["batches"].items():
                print(f"批次 {batch_id}: 作业{batch_info['jobs']}, "
                      f"派遣时间: {batch_info['dispatch_time']}")
                if batch_info['served_requirements']:
                    print(f"  服务需求: {batch_info['served_requirements']}")
            
            print("\n=== 交付短缺 ===")
            total_shortage = 0
            for req_id, shortage in result["delivery_shortfalls"].items():
                if shortage > 0:
                    print(f"需求 {req_id}: 短缺量 {shortage}")
                    total_shortage += shortage
            if total_shortage == 0:
                print("无交付短缺")



if __name__ == "__main__":
    # 定义 runner.py 中的小规模算例配置
    small_configs = [
        (40, 10, 20, 5),
        # (40, 10, 20, 10),
        # (40, 10, 20, 15),
        # (60, 10, 40, 5),
        # (60, 10, 40, 10),
        # (60, 10, 40, 15),
        # (80, 10, 60, 5),
        # (80, 10, 60, 10),
        # (80, 10, 60, 15),
        # (100, 10, 80, 5),
        # (100, 10, 80, 10),
        # (100, 10, 80, 15),
    ]

    for initial_jobs, machines, dynamic_jobs, distributors in small_configs:
        print("\n" + "="*60)
        print(f"🚀 开始运行算例: {initial_jobs}初始作业, {machines}机器, {dynamic_jobs}动态作业, {distributors}配送商")
        print("="*60)

        # 1. 创建配置实例并设置参数
        config = Config()
        config.num_initial_jobs = initial_jobs
        config.num_machines = machines
        config.num_distributors = distributors
        config.num_dynamic_jobs = dynamic_jobs
        
        # 使用 runner.py 中的标准参数
        config.min_operations = 1
        config.max_operations = 3
        config.min_processing_time = 5
        config.max_processing_time = 15
        config.min_job_amount = 50
        config.max_job_amount = 100
        config.min_delivery_requirements = 2
        config.max_delivery_requirements = 3
        config.earliest_delivery_time = 10
        config.latest_delivery_time = 200
        config.batch_arrival_probability = 0.5
        config.max_batch_size = 5

        # 2. 创建场景实例
        scenario = FlexibleJobShopScenario(config=config)
        
        # 3. 创建数学模型
        model = IFJSSPDPModel(scenario, max_batches=8)
        
        # 4. 设置目标函数
        model.set_objective("weighted")
        
        # 5. 求解
        result = model.solve(time_limit=config.solver_time_limit) # 可以增加时间限制以获得更好的解
        
        # 6. 打印结果
        model.print_solution(result)