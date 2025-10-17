"""
粒子群优化 (Particle Swarm Optimization)
"""
import random
import numpy as np
import copy
from ..base_algorithm import BaseAlgorithm

class PSO_Agent(BaseAlgorithm):
    """
    使用粒子群优化算法求解 FJSP 的智能体。
    注意：这是一个简化的实现，用于演示目的。
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        # 从配置中加载超参数
        self.max_iterations = self.config.get('max_iterations', 50)
        self.num_particles = self.config.get('num_particles', 20)
        self.w = self.config.get('w', 0.5)
        self.c1 = self.config.get('c1', 1.5)
        self.c2 = self.config.get('c2', 1.5)

    def _decode_and_evaluate(self, op_sequence):
        """
        解码工序序列为一个完整的调度解，并评估其目标函数值。
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

    def solve(self):
        print(f"Running {self.__class__.__name__}...")

        # 初始化粒子群
        particles = []
        for _ in range(self.num_particles):
            # 位置（工序序列）
            position = list(range(self.n_ops))
            random.shuffle(position)
            # 速度（交换序列）
            velocity = [] 
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position,
                'best_fitness': float('inf')
            })

        g_best_position = None
        g_best_fitness = float('inf')

        for i in range(self.max_iterations):
            for particle in particles:
                fitness, _, _ = self._decode_and_evaluate(particle['position'])
                
                # 更新个体最优
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position']
                
                # 更新全局最优
                if fitness < g_best_fitness:
                    g_best_fitness = fitness
                    g_best_position = particle['position']

            # 更新粒子速度和位置
            for particle in particles:
                # --- 更新速度 ---
                # 这是一个简化的PSO，速度由交换序列表示
                # v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x)
                
                # 惯性部分
                new_velocity = particle['velocity'][:int(self.w * len(particle['velocity']))]
                
                # 个体最优部分
                r1 = random.random()
                if r1 < self.c1:
                    # 找到从当前位置到个体最优位置的交换
                    # (这是一个简化的启发式)
                    for j in range(self.n_ops):
                        if particle['position'][j] != particle['best_position'][j]:
                            swap_with_idx = particle['position'].index(particle['best_position'][j])
                            new_velocity.append((j, swap_with_idx))
                            # 应用一次交换以模拟移动
                            p_pos = list(particle['position'])
                            p_pos[j], p_pos[swap_with_idx] = p_pos[swap_with_idx], p_pos[j]
                            particle['position'] = p_pos
                            break # 简化，每次只做一个交换

                # 全局最优部分
                r2 = random.random()
                if r2 < self.c2:
                    # 找到从当前位置到全局最优位置的交换
                    for j in range(self.n_ops):
                        if particle['position'][j] != g_best_position[j]:
                            swap_with_idx = particle['position'].index(g_best_position[j])
                            new_velocity.append((j, swap_with_idx))
                            # 应用一次交换
                            p_pos = list(particle['position'])
                            p_pos[j], p_pos[swap_with_idx] = p_pos[swap_with_idx], p_pos[j]
                            particle['position'] = p_pos
                            break

                particle['velocity'] = new_velocity
                
                # --- 更新位置 ---
                # 应用速度（交换序列）
                for j1, j2 in particle['velocity']:
                    p_pos = list(particle['position'])
                    p_pos[j1], p_pos[j2] = p_pos[j2], p_pos[j1]
                    particle['position'] = p_pos

            if (i + 1) % 20 == 0:
                print(f"  PSO | Iteration {i + 1}/{self.max_iterations} | Best Objective: {g_best_fitness:.2f}")

        _, best_solution, best_results = self._decode_and_evaluate(g_best_position)
        
        print(f"{self.__class__.__name__} finished.")
        return best_solution, best_results
