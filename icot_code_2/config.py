"""
中心配置文件
Central Configuration File
"""
import os
import multiprocessing

# --- 路径设置 (Path Settings) ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# --- 实验设置 (Experiment Settings) ---
# 用于并行处理的 CPU 核心数，建议不要超过物理核心数
# Number of CPU cores for parallel processing. It's recommended not to exceed the number of physical cores.
NUM_PROCESSES = 1

# --- 实例设置 (Instance Settings) ---
# glob模式，用于查找要运行的实例文件
# Glob pattern to find instance files to run.
INSTANCE_GLOB_PATTERN = "generated_*.json"

# --- 结果设置 (Result Settings) ---
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
RESULTS_CSV_FILENAME = 'all_results.csv'
ANALYSIS_FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# --- 算法超参数设置 (Algorithm Hyperparameter Settings) ---
# 为了快速演示和调试，所有参数都设置得较低
# For quick demonstration and debugging, all parameters are set to low values.

# 深度强化学习 (Deep Reinforcement Learning)
DQN_CONFIG = {
    "episodes": 500,
    "batch_size": 256,
    "gamma": 0.99,
    "eps_start": 0.99,
    "eps_end": 0.001,
    "eps_decay": 3000,  # 较快的衰减 for demo
    "lr": 1e-4,
    "memory_size": 50000
}

A2C_CONFIG = {
    "is_implemented": False
    # A2C-specific params here
}

PPO_CONFIG = {
    "is_implemented": False
    # PPO-specific params here
}


# 元启发式算法 (Metaheuristic Algorithms)
GA_CONFIG = {
    "generations": 50,
    "population_size": 30,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1
}

SA_CONFIG = {
    "max_iterations": 1000,
    "initial_temp": 1000,
    "alpha": 0.99  # 冷却率
}

PSO_CONFIG = {
    "max_iterations": 50,
    "num_particles": 20,
    "w": 0.5,      # 惯性权重
    "c1": 1.5,     # 认知系数
    "c2": 1.5      # 社会系数
}

# --- CSV文件字段名 (CSV File Fieldnames) ---
CSV_FIELDNAMES = [
    "instance", "algorithm", "objective_value", "makespan",
    "total_transportation_time", "total_energy", "total_carbon",
    "is_feasible", "execution_time_s", "max_wip"
]
