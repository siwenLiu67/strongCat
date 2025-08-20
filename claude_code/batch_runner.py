"""
通用算法运行框架
支持单个算法在多个算例上的批量运行
根据论文Table 2参数设置生成标准算例
"""

import sys
import os
import time
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# 添加当前目录到 Python 路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


from config import Config
from case_generator import FlexibleJobShopScenario
from algorithm_results_saver import save_algorithm_results_csv, set_random_seed
from run_ruleDqn_dispatchHeuri import run_ruleDqn_dispatchHeuri_experiment

@dataclass
class PaperInstanceConfig:
    """基于论文参数的算例配置"""
    num_machines: int
    num_initial_jobs: int
    num_dynamic_jobs: int
    num_distributors: int
    name: str = ""
    
    @property
    def total_jobs(self):
        return self.num_initial_jobs + self.num_dynamic_jobs
    
    def __post_init__(self):
        if not self.name:
            if self.num_initial_jobs <= 30:
                size = "Small"
            elif self.num_initial_jobs <= 100:
                size = "Medium"
            else:
                size = "Large"
            self.name = f"{size}-{self.num_initial_jobs + self.num_dynamic_jobs}J{self.num_machines}M{self.num_distributors}D"

class PaperBasedInstanceGenerator:
    """基于论文参数生成标准算例"""
    
    def generate_all_paper_instances(self) -> List[PaperInstanceConfig]:
        """生成所有论文参数组合的算例（基于Table 4的36个实验算例）"""
        instances = []
        
        # 基于论文Table 4的参数设置
        # 小规模算例 (40-100作业): 初始作业20, 机器10
        small_configs = [
            # 动态作业20, 配送商[5, 10, 15]
            (20, 10, 20, 5),
            # (20, 10, 20, 10),
            # (20, 10, 20, 15),
            # # 动态作业40, 配送商[5, 10, 15]
            # (20, 10, 40, 5),
            # (20, 10, 40, 10),
            # (20, 10, 40, 15),
            # # 动态作业60, 配送商[5, 10, 15]
            # (20, 10, 60, 5),
            # (20, 10, 60, 10),
            # (20, 10, 60, 15),
            # # 动态作业80, 配送商[5, 10, 15]
            # (20, 10, 80, 5),
            # (20, 10, 80, 10),
            # (20, 10, 80, 15),
        ]
        
        # 中规模算例 (100-160作业): 初始作业80, 机器20
        # medium_configs = [
        #     # 动态作业20, 配送商[5, 10, 15]
        #     (80, 20, 20, 5),
        #     (80, 20, 20, 10),
        #     (80, 20, 20, 15),
        #     # 动态作业40, 配送商[5, 10, 15]
        #     (80, 20, 40, 5),
        #     (80, 20, 40, 10),
        #     (80, 20, 40, 15),
        #     # 动态作业60, 配送商[5, 10, 15]
        #     (80, 20, 60, 5),
        #     (80, 20, 60, 10),
        #     (80, 20, 60, 15),
        #     # 动态作业80, 配送商[5, 10, 15]
        #     (80, 20, 80, 5),
        #     (80, 20, 80, 10),
        #     (80, 20, 80, 15),
        # ]
        
        # # 大规模算例 (160-220作业): 初始作业140, 机器30
        # large_configs = [
        #     # 动态作业20, 配送商[5, 10, 15]
        #     (140, 30, 20, 5),
        #     (140, 30, 20, 10),
        #     (140, 30, 20, 15),
        #     # 动态作业40, 配送商[5, 10, 15]
        #     (140, 30, 40, 5),
        #     (140, 30, 40, 10),
        #     (140, 30, 40, 15),
        #     # 动态作业60, 配送商[5, 10, 15]
        #     (140, 30, 60, 5),
        #     (140, 30, 60, 10),
        #     (140, 30, 60, 15),
        #     # 动态作业80, 配送商[5, 10, 15]
        #     (140, 30, 80, 5),
        #     (140, 30, 80, 10),
        #     (140, 30, 80, 15),
        # ]
        
        # 生成所有算例
        for config in small_configs :#+ medium_configs + large_configs:
            initial_jobs, machines, dynamic_jobs, distributors = config
            instance = PaperInstanceConfig(
                num_machines=machines,
                num_initial_jobs=initial_jobs,
                num_dynamic_jobs=dynamic_jobs,
                num_distributors=distributors
            )
            instances.append(instance)
        
        return instances
    
    def generate_benchmark_instances(self) -> List[PaperInstanceConfig]:
        """生成基准测试算例（论文中的关键配置）"""
        return [
            # 基准配置 - 对应论文中的典型设置
           # PaperInstanceConfig(20, 20, 50, 5),   # 标准配置
            PaperInstanceConfig(4, 4, 2, 1),   # 资源受限
           # PaperInstanceConfig(30, 20, 70, 7),   # 大规模配置
        ]

class UniversalAlgorithmRunner:
    """增强的通用算法运行器"""
    
    def __init__(self):
        self.algorithm_modules = {
            # 强化学习算法
            'DQN': 'dqn_model',
            'SARSA': 'sarsa_model', 
            'PPO': 'ppo_model',
            'A3C': 'a3c_model',
            'DDPG': 'ddpg_model',
            'HRL_GAT': 'hrl_gnn_model',
            'RuleDQN_DispatchHeuri': 'run_ruleDqn_dispatchHeuri',
            
            # 启发式算法
            'GA': 'heuristics_model',
            'PSO': 'pso_model',
            'NSGA2': 'nsga2_model',
            'Greedy': 'greedy_model',
            'Random': 'random_model',
        }
        
        # 论文算例生成器
        self.paper_generator = PaperBasedInstanceGenerator()
        
        # 默认种子列表
        self.default_seeds = [42, 123, 456, 789, 999]
    
    def get_algorithm_function(self, algo_name: str):
        """根据算法名称获取对应的运行函数"""
        if algo_name not in self.algorithm_modules:
            raise ValueError(f"不支持的算法: {algo_name}. 支持的算法: {list(self.algorithm_modules.keys())}")
        
        module_name = self.algorithm_modules[algo_name]
        
        try:
            module = importlib.import_module(module_name)
            
            # 根据算法类型确定函数名
            if algo_name in ['DQN', 'SARSA', 'PPO', 'A3C', 'DDPG']:
                func_name = f'run_{algo_name.lower()}_experiment'
            elif algo_name == 'HRL_GAT':
                func_name = 'run_hrl_gat_experiment'
            elif algo_name == 'RuleDQN_DispatchHeuri':
                func_name = 'main'
            elif algo_name in ['GA', 'PSO', 'NSGA2', 'Greedy', 'Random']:
                func_name = f'run_{algo_name.lower()}_experiment'
            else:
                func_name = 'main'
            
            if hasattr(module, func_name):
                return getattr(module, func_name)
            else:
                # 尝试常见的函数名
                for name in ['main', 'run_experiment', f'run_{algo_name.lower()}']:
                    if hasattr(module, name):
                        return getattr(module, name)
                
                raise AttributeError(f"模块 {module_name} 中没有找到合适的运行函数")
                
        except ImportError as e:
            raise ImportError(f"无法导入算法模块 {module_name}: {e}")
    
    def create_paper_config(self, instance: PaperInstanceConfig, base_config: Optional[Config] = None) -> Config:
        """基于论文参数创建配置"""
        if base_config is None:
            config = Config()
        else:
            # 复制基础配置
            config = base_config
            
        # 设置正确的作业数量
        config.num_initial_jobs = instance.num_initial_jobs  # 初始作业数量，用于生成算例
        config.num_machines = instance.num_machines
        config.num_distributors = instance.num_distributors
        
        # 动态作业相关
        config.num_dynamic_jobs = instance.num_dynamic_jobs
        
        # 论文Table 2中的参数
        config.min_operations = 1
        config.max_operations = 3
        config.min_processing_time = 5
        config.max_processing_time = 15
        config.min_job_amount = 50
        config.max_job_amount = 100
        
        # 配送相关参数
        config.min_delivery_requirements = 2
        config.max_delivery_requirements = 3
        config.earliest_delivery_time = 10
        config.latest_delivery_time = 200  # 根据处理时间估算
        
        # 动态到达参数
        config.batch_arrival_probability = 0.5
        config.max_batch_size = 5
        
        return config
    
    def run_single_experiment(self, algo_name: str, instance: PaperInstanceConfig, seed: int, 
                            base_config: Optional[Config] = None, **kwargs) -> Dict:
        """运行单个实验"""
        print(f"🚀 开始运行: {algo_name} - {instance.name} - seed{seed}")
        print(f"   参数: {instance.total_jobs}作业({instance.num_initial_jobs}+{instance.num_dynamic_jobs}), "
              f"{instance.num_machines}机器, {instance.num_distributors}配送商")
        
        # 设置随机种子
        set_random_seed(seed)
        
        # 创建配置
        config = self.create_paper_config(instance, base_config)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 生成算例
            case = FlexibleJobShopScenario(config=config)
            
            # 获取算法函数
            algorithm_func = self.get_algorithm_function(algo_name)
            
            # 运行算法
            if algo_name == 'RuleDQN_DispatchHeuri':
                result = run_ruleDqn_dispatchHeuri_experiment(config, case, seed)
            else:
                result = algorithm_func(config, case, seed, **kwargs)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 保存结果
            instance_id = instance.name
            success = save_algorithm_results_csv(
                algo_name=algo_name,
                instance_id=instance_id,
                seed=seed,
                config=config,
                stats=result.get('stats', {}),
                env=result.get('env', case),
                total_time=total_time,
                additional_metrics=result.get('additional_metrics', {})
            )
            
            print(f"✅ 完成: {algo_name} - {instance.name} (耗时: {total_time:.2f}s)")
            
            return {
                'success': success,
                'result': result,
                'total_time': total_time,
                'instance_id': instance_id
            }
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"❌ 失败: {algo_name} - {instance.name}: {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'total_time': total_time,
                'instance_id': instance.name
            }
    

    def run_paper_experiments(self, algo_name: str, 
                            instance_type: str,
                            base_config,
                            **kwargs):
        """运行基于论文参数的实验"""
        
        # 选择算例类型
        if instance_type == 'all':
            instances = self.paper_generator.generate_all_paper_instances()
        
        elif instance_type == 'benchmark':
            instances = self.paper_generator.generate_benchmark_instances()
        else:
            raise ValueError(f"不支持的算例类型: {instance_type}")

        seeds = base_config.seeds

        print(f"📊 准备运行 {algo_name} 算法")
        print(f"   算例类型: {instance_type}")
        print(f"   算例数量: {len(instances)}")
        print(f"   种子数量: {len(seeds)}")
        print(f"   总实验数: {len(instances) * len(seeds)}")
        print("="*60)
        
        # 显示算例详情
        print("算例详情:")
        for i, instance in enumerate(instances, 1):
            print(f"  {i:2d}. {instance.name}: "
                  f"{instance.total_jobs}作业({instance.num_initial_jobs}+{instance.num_dynamic_jobs}), "
                  f"{instance.num_machines}机器, {instance.num_distributors}配送商")
        print("="*60)
        
        results = []
        total_experiments = len(instances) * len(seeds)
        completed = 0
        
        for instance in instances:
            for seed in seeds:
                result = self.run_single_experiment(
                    algo_name=algo_name,
                    instance=instance,
                    seed=seed,
                    base_config=base_config,
                    **kwargs
                )
                
                results.append(result)
                completed += 1
                
                # 显示进度
                progress = (completed / total_experiments) * 100
                print(f"📈 进度: {completed}/{total_experiments} ({progress:.1f}%)")
                print("-" * 40)
        
        # 统计结果
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        total_time = sum(r['total_time'] for r in results)
        
        print("="*60)
        print(f"🎯 批量实验完成!")
        print(f"   成功: {successful}/{total_experiments}")
        print(f"   失败: {failed}/{total_experiments}")
        print(f"   总耗时: {total_time:.2f}s")
        print(f"   平均耗时: {total_time/total_experiments:.2f}s")
        
        return results

def run_batch_experiment(
    algorithm,
    instance_type
):
    runner = UniversalAlgorithmRunner()
    base_config = Config()

    results = runner.run_paper_experiments(
        algo_name=algorithm,
        instance_type=instance_type,
        base_config=base_config
    )
    successful = sum(1 for r in results if r['success'])
    print(f"\n📄 实验摘要:")
    print(f"   总实验数: {len(results)}")
    print(f"   成功: {successful}")
    print(f"   失败: {len(results) - successful}")
    print(f"   结果已保存到: results/algorithm_comparison.csv")
    return results

if __name__ == "__main__":
    # 直接调用，无需命令行参数
    run_batch_experiment(
        algorithm="RuleDQN_DispatchHeuri",   # 修改为你要运行的法
        instance_type="all", # 可选: benchmark, all
    )
