"""
灵活作业车间调度问题 (FJSP) 的仿真环境。
Simulation Environment for the Flexible Job Shop Scheduling Problem (FJSP).
"""
import json
import numpy as np

class FjspEnv:
    """
    FJSP 环境类。
    - 加载实例文件。
    - 提供评估一个完整调度解（solution）的功能。
    """
    def __init__(self, instance_path):
        """
        :param instance_path: str, a path to the instance JSON file.
        """
        self.instance_path = instance_path
        self._load_instance()

    def _load_instance(self):
        """
        从 JSON 文件加载实例数据并进行预处理。
        """
        with open(self.instance_path, 'r') as f:
            data = json.load(f)

        # 基本信息
        self.num_jobs = data['num_jobs']
        self.num_machines = data['num_machines']
        self.num_transporters = data['num_transporters']
        self.transporters = data['transporters']  # 加载运输工具详细信息
        
        # 约束
        self.Qmax = data['constraints']['max_wip']
        self.Emax = data['constraints']['total_energy']
        self.Cmax = data['constraints']['total_carbon']
        
        # 权重
        self.weights = data['weights']

        # 作业和工序
        self.jobs = data['jobs']
        
        # --- 预处理工序数据 ---
        self.operations = []
        self.job_op_to_global_op_map = {} # (job_id, op_id) -> global_op_idx
        op_idx_counter = 0
        for job in self.jobs:
            for op in job['operations']:
                op_data = {
                    'job_id': job['job_id'],
                    'op_id': op['op_id'],
                    'proc_options': []
                }
                for option in op['processing_options']:
                    op_data['proc_options'].append(
                        (option['machine'], option['processing_time'], option['energy_consumption'], option['carbon_emission'])
                    )
                self.operations.append(op_data)
                self.job_op_to_global_op_map[(job['job_id'], op['op_id'])] = op_idx_counter
                op_idx_counter += 1
        
        self.num_total_ops = len(self.operations)
        print(f"  - Loaded Instance: {self.instance_path}")
        print(f"  - Jobs: {self.num_jobs}, Machines: {self.num_machines}, Operations: {self.num_total_ops}")


    def evaluate_solution(self, solution):
        """
        评估一个给定的完整调度解。
        
        :param solution: dict, a dictionary describing the complete schedule.
            - "op_sequence": list of global operation indices.
            - "machine_assignment": list where index is global op_idx and value is machine_idx.
            - "transport_assignment": list where index is job_idx and value is transporter_idx.
        :return: dict, a dictionary containing all performance metrics.
        """
        op_sequence = solution.get("op_sequence", [])
        machine_assignment = solution.get("machine_assignment", [])
        transport_assignment = solution.get("transport_assignment", [])

        if not op_sequence or not machine_assignment or not transport_assignment or len(machine_assignment) != self.num_total_ops or len(transport_assignment) != self.num_jobs:
            # 返回一个表示无效解的默认结果
            return {
                "objective_value": None, "makespan": None,
                "total_transport_time": None, "total_energy": None,
                "total_carbon": None, "max_wip": None, "is_feasible": False
            }

        machine_release_times = np.zeros(self.num_machines)
        op_completion_times = {}  # (job_id, op_id) -> time
        
        total_energy = 0
        total_carbon = 0

        for op_idx in op_sequence:
            op_info = self.operations[op_idx]
            job_id, op_id = op_info['job_id'], op_info['op_id']
            
            assigned_machine = machine_assignment[op_idx]
            
            # 获取该工序在指定机器上的处理信息
            proc_time, energy, carbon = -1, -1, -1
            for m_idx, pt, e, c in op_info['proc_options']:
                if m_idx == assigned_machine:
                    proc_time, energy, carbon = pt, e, c
                    break
            
            if proc_time == -1: # 如果分配了无效的机器
                return {
                    "objective_value": None, "makespan": None,
                    "total_transport_time": None, "total_energy": None,
                    "total_carbon": None, "max_wip": None, "is_feasible": False
                }

            # 计算开始时间
            # 找到该作业所有先前工序的最大完成时间
            precedent_completion_time = 0
            if op_id > 0:
                precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)

            start_time = max(machine_release_times[assigned_machine], precedent_completion_time)
            end_time = start_time + proc_time
            
            # 更新状态
            machine_release_times[assigned_machine] = end_time
            op_completion_times[(job_id, op_id)] = end_time
            total_energy += energy
            total_carbon += carbon

        # --- 计算所有指标 ---
        # 1. 计算每个作业的最终工序完成时间
        job_op_completion_times = {i: 0 for i in range(self.num_jobs)}
        for (job_id, op_id), time in op_completion_times.items():
            if time > job_op_completion_times[job_id]:
                job_op_completion_times[job_id] = time

        # 2. 计算包含运输时间的每个作业的最终完成时间
        job_final_completion_times = []
        total_transport_time = 0
        transport_carbon = 0
        for job_id, transporter_idx in enumerate(transport_assignment):
            transporter = self.transporters[transporter_idx]
            transport_time = transporter['transport_time']
            carbon_factor = transporter['carbon_factor']
            
            # 作业的最终完成时间 = 工序完成时间 + 运输时间
            final_time = job_op_completion_times[job_id] + transport_time
            job_final_completion_times.append(final_time)
            
            total_transport_time += transport_time
            transport_carbon += transport_time * carbon_factor

        # 3. Makespan 是所有作业最终完成时间的最大值
        makespan = np.max(job_final_completion_times) if job_final_completion_times else 0
        
        # 4. 总碳排放 = 加工碳排放 + 运输碳排放
        total_carbon += transport_carbon
        
        # 简化：在制品 (WIP) 暂时设为0
        max_wip = 0

        # 5. 检查可行性
        is_feasible = (
            max_wip <= self.Qmax and
            total_energy <= self.Emax and
            total_carbon <= self.Cmax
        )
        
        # 6. 目标函数：以 Makespan 为主，对不可行解施加巨大惩罚
        objective_value = makespan
        if not is_feasible:
            objective_value += 1e9  # Penalty for infeasible solutions

        return {
            "objective_value": round(objective_value, 2),
            "makespan": round(makespan, 2),
            "total_transportation_time": round(total_transport_time, 2),
            "total_energy": round(total_energy, 2),
            "total_carbon": round(total_carbon, 2),
            "max_wip": max_wip,
            "is_feasible": is_feasible
        }
