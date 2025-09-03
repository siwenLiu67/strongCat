"""
测试元启发式算法环境适配器和遗传算法启发式求解器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metaheuristic_environment_adapter import MetaheuristicEnvironment, create_metaheuristic_algorithm
from genetic_algorithm_heuristic import GeneticAlgorithmHeuristicSolver
from case_generator import FlexibleJobShopScenario
from config import Config


def test_metaheuristic_environment():
    """测试元启发式算法环境适配器"""
    print("测试元启发式算法环境适配器...")
    
    # 创建配置对象并设置参数
    config = Config()
    config.num_initial_jobs = 5
    config.num_machines = 3
    config.num_distributors = 2
    
    # 创建测试算例
    scenario = FlexibleJobShopScenario(config=config)
    
    # 测试遗传算法
    try:
        env = MetaheuristicEnvironment(scenario, "genetic", {
            "population_size": 30,
            "generations": 50,
            "fitness_function": "total_weighted_tardiness"
        })
        state = env.reset()
        print("✓ 遗传算法环境创建成功")
    except Exception as e:
        print(f"✗ 遗传算法环境创建失败: {e}")
        return False
    
    # 测试模拟退火算法
    try:
        env = MetaheuristicEnvironment(scenario, "simulated_annealing", {
            "initial_temperature": 500,
            "cooling_rate": 0.9,
            "fitness_function": "makespan"
        })
        state = env.reset()
        print("✓ 模拟退火算法环境创建成功")
    except Exception as e:
        print(f"✗ 模拟退火算法环境创建失败: {e}")
        return False
    
    # 测试算法工厂函数
    try:
        algorithms = [
            create_metaheuristic_algorithm("genetic", population_size=20),
            create_metaheuristic_algorithm("simulated_annealing", initial_temperature=1000),
            create_metaheuristic_algorithm("tabu_search", tabu_tenure=5),
            create_metaheuristic_algorithm("pso", swarm_size=25)
        ]
        print("✓ 所有算法工厂函数创建成功")
    except Exception as e:
        print(f"✗ 算法工厂函数创建失败: {e}")
        return False
    
    return True


def test_genetic_algorithm_heuristic():
    """测试遗传算法启发式求解器"""
    print("\n测试遗传算法启发式求解器...")
    
    # 创建配置对象并设置参数
    config = Config()
    config.num_initial_jobs = 5
    config.num_machines = 3
    config.num_distributors = 2
    
    # 创建测试算例
    scenario = FlexibleJobShopScenario(config=config)
    
    try:
        # 创建环境实例
        from environment import WarehouseEnvironment
        env = WarehouseEnvironment(config, scenario)
        
        # 测试遗传算法求解器
        solver = GeneticAlgorithmHeuristicSolver(
            env,
            config,
            population_size=20,
            generations=30,
            crossover_rate=0.8,
            mutation_rate=0.1,
            selection_method="tournament",
            fitness_function="total_weighted_tardiness"
        )
        print("✓ 遗传算法求解器创建成功")
        
        # 测试批量实验接口
        result = run_genetic_algorithm_experiment(config, scenario, seed=42, 
                                                 population_size=20, generations=30)
        print("✓ 批量实验接口测试成功")
        
    except Exception as e:
        print(f"✗ 遗传算法求解器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def run_genetic_algorithm_experiment(config, case, seed, **kwargs):
    """简化的批量实验函数"""
    from genetic_algorithm_heuristic import GeneticAlgorithmHeuristicSolver
    
    # 创建环境实例
    from environment import WarehouseEnvironment
    env = WarehouseEnvironment(config, case)
    
    # 创建求解器
    solver = GeneticAlgorithmHeuristicSolver(
        env, config,
        population_size=kwargs.get('population_size', 20),
        generations=kwargs.get('generations', 30),
        crossover_rate=kwargs.get('crossover_rate', 0.8),
        mutation_rate=kwargs.get('mutation_rate', 0.1),
        selection_method=kwargs.get('selection_method', 'tournament'),
        fitness_function=kwargs.get('fitness_function', 'total_weighted_tardiness')
    )
    
    # 求解并返回结果
    return solver.solve()


if __name__ == "__main__":
    print("开始测试元启发式算法...")
    
    success1 = test_metaheuristic_environment()
    success2 = test_genetic_algorithm_heuristic()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！元启发式算法框架工作正常")
    else:
        print("\n❌ 部分测试失败，请检查代码")
