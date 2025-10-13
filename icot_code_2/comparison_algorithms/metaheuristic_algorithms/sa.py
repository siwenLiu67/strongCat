"""
模拟退火 (Simulated Annealing)
"""
import random
import numpy as np
import math
import copy
from ..base_algorithm import BaseAlgorithm

class SA_Agent(BaseAlgorithm):
    """
    使用模拟退火算法求解 FJSP 的智能体。
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        # 从配置中加载超参数
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.temp = self.config.get('initial_temp', 1000)
        self.alpha = self.config.get('alpha', 0.99)

    def _generate_initial_solution(self):
        """生成一个随机的初始解（工序序列）。"""
        op_sequence = list(range(self.n_ops))
        random.shuffle(op_sequence)
        return op_sequence

    def _decode_and_evaluate(self, op_sequence):
        """
        解码工序序列为一个完整的调度解，并评估其目标函数值。
        使用简单的启发式规则分配机器：选择最早可以开始该工序的机器。
        """
        machine_release_times = np.zeros(self.n_machines)
        op_completion_times = {}
        machine_assignment = [-1] * self.n_ops

        for op_idx in op_sequence:
            op_info = self.env.operations[op_idx]
            job_id, op_id = op_info['job_id'], op_info['op_id']
            
            precedent_completion_time = op_completion_times.get((job_id, op_id - 1), 0)
            
            best_machine = -1
            min_end_time = float('inf')
            
            for m_idx, proc_time, _, _ in op_info['proc_options']:
                start_time = max(machine_release_times[m_idx], precedent_completion_time)
                end_time = start_time + proc_time
                if end_time < min_end_time:
                    min_end_time = end_time
                    best_machine = m_idx
            
            machine_release_times[best_machine] = min_end_time
            op_completion_times[(job_id, op_id)] = min_end_time
            machine_assignment[op_idx] = best_machine

        transport_assignment = [random.randint(0, self.env.num_transporters - 1) for _ in range(self.n_jobs)]
        
        solution = {
            "op_sequence": op_sequence,
            "machine_assignment": machine_assignment,
            "transport_assignment": transport_assignment
        }
        
        results = self.env.evaluate_solution(solution)
        return results['objective_value'], solution, results

    def _get_neighbor(self, op_sequence):
        """
        通过交换两个随机工序的位置来获取一个邻域解。
        """
        neighbor = op_sequence[:]
        idx1, idx2 = random.sample(range(len(neighbor)), 2)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        return neighbor

    def solve(self):
        print(f"Running {self.__class__.__name__}...")
        
        # 生成初始解
        current_solution_seq = self._generate_initial_solution()
        current_objective, _, _ = self._decode_and_evaluate(current_solution_seq)
        
        best_solution_seq = current_solution_seq
        best_objective = current_objective
        
        for i in range(self.max_iterations):
            # 生成邻域解
            neighbor_seq = self._get_neighbor(current_solution_seq)
            neighbor_objective, _, _ = self._decode_and_evaluate(neighbor_seq)
            
            # 计算能量差
            delta = neighbor_objective - current_objective
            
            # 决定是否接受新解
            if delta < 0 or random.random() < math.exp(-delta / self.temp):
                current_solution_seq = neighbor_seq
                current_objective = neighbor_objective
            
            # 更新全局最优解
            if current_objective < best_objective:
                best_solution_seq = current_solution_seq
                best_objective = current_objective
            
            # 降温
            self.temp *= self.alpha

            if (i + 1) % 100 == 0:
                print(f"  SA | Iteration {i + 1}/{self.max_iterations} | Best Objective: {best_objective:.2f} | Current Temp: {self.temp:.2f}")

        # 解码最终的最优序列以获得完整解
        _, best_solution, best_results = self._decode_and_evaluate(best_solution_seq)
        
        print(f"{self.__class__.__name__} finished.")
        return best_solution, best_results
