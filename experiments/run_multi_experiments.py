"""
批量运行多算例的实验框架
支持在多个不同规模算例上运行各种算法
提供通用的实验配置和运行管理功能
"""

import sys
import os
import time
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

# 添加项目根目录到路径
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.config import Config
from instances.case_generator import FlexibleJobShopScenario
from results.algorithm_results_saver import (
    save_algorithm_results_csv,
    set_random_seed,
    generate_instance_id
)

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    num_machines: int
    num_initial_jobs: int
    num_dynamic_jobs: int
    num_distributors: int
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            if self.num_initial_jobs <= 30:
                size = "Small"
            elif self.num_initial_jobs <= 100:
                size = "Medium"
            else:
                size = "Large"
            self.name = f"{size}-{self.num_initial_jobs + self.num_dynamic_jobs}J{self.num_machines}M{self.num_distributors}D"
    
    @property
    def total_jobs(self):
        return self.num_initial_jobs + self.num_dynamic_jobs

class MultiExperimentRunner:
    """多算例批量实验运行器"""
    
    def __init__(self, base_config: Optional[Config] = None):
        self.base_config = base_config or Config()
        
        # 默认种子列表
        self.seeds = [42, 123, 456, 789, 999]
        
        # 算例配置列表
        self.experiment_configs = self._generate_experiment_configs()
    
    def _generate_experiment_configs(self) -> List[ExperimentConfig]:
        """生成实验配置列表"""
        configs = []
        
        # 小规模算例
        small_configs = [
            # (初始作业数, 机器数, 动态作业数, 配送商数)
            (20, 10, 20, 5),
            (20, 10, 20, 10),
            (20, 10, 40, 5),
            (20, 10, 40, 10)
        ]
        
        # 中规模算例
        medium_configs = [
            (80, 20, 20, 5),
            (80, 20, 20, 10),
            (80, 20, 40, 5),
            (80, 20, 40, 10)
        ]
        
        # 生成所有实验配置
        for config in small_configs + medium_configs:
            initial_jobs, machines, dynamic_jobs, distributors = config
            experiment_config = ExperimentConfig(
                num_machines=machines,
                num_initial_jobs=initial_jobs,
                num_dynamic_jobs=dynamic_jobs,
                num_distributors=distributors
            )
            configs.append(experiment_config)
        
        return configs
    
    def create_experiment_config(self, exp_config: ExperimentConfig) -> Config:
        """基于实验配置创建具体的运行配置"""
        config = self.base_config
        
        # 设置基本参数
        config.num_initial_jobs = exp_config.num_initial_jobs
        config.num_machines = exp_config.num_machines
        config.num_distributors = exp_config.num_distributors
        config.num_dynamic_jobs = exp_config.num_dynamic_jobs
        
        # 设置通用参数
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
        config.latest_delivery_time = 200
        
        # 动态到达参数
        config.batch_arrival_probability = 0.5
        config.max_batch_size = 5
        
        return config
    
    def get_algorithm_module(self, algo_name: str) -> Any:
        """
        动态导入算法模块
        """
        try:
            module = importlib.import_module(f"{algo_name.lower()}_model")
            return module
        except ImportError:
            try:
                module = importlib.import_module(f"run_{algo_name.lower()}")
                return module
            except ImportError as e:
                raise ImportError(f"无法导入算法模块 {algo_name}: {e}")

    def run_single_experiment(self, algo_name: str, exp_config: ExperimentConfig, 
                            seed: int) -> Dict[str, Any]:
        """运行单个实验"""
        print(f"\n开始运行实验: {algo_name} - {exp_config.name} (seed={seed})")
        print(f"配置: {exp_config.total_jobs}作业({exp_config.num_initial_jobs}+{exp_config.num_dynamic_jobs}), "
              f"{exp_config.num_machines}机器, {exp_config.num_distributors}配送商")
        
        try:
            # 设置随机种子
            set_random_seed(seed)
            
            # 创建实验配置
            config = self.create_experiment_config(exp_config)
            config.seeds = [seed]
            
            # 生成算例
            case = FlexibleJobShopScenario(config=config)
            
            # 记录开始时间
            start_time = time.time()
            
            # 获取算法模块并运行
            algorithm = self.get_algorithm_module(algo_name)
            result = algorithm.run_experiment(config, case, seed)
            
            # 计算总时间
            total_time = time.time() - start_time
            
            # 保存结果
            instance_id = exp_config.name
            success = save_algorithm_results_csv(
                algo_name=algo_name,
                instance_id=instance_id,
                seed=seed,
                config=config,
                stats=result.get('stats', {}),
                env=result.get('env', None),
                total_time=total_time,
                additional_metrics=result.get('additional_metrics', {})
            )
            
            print(f"实验完成: {algo_name} - {exp_config.name} (耗时: {total_time:.2f}s)")
            
            return {
                'success': True,
                'stats': result.get('stats', {}),
                'total_time': total_time,
                'instance_id': instance_id,
                'additional_metrics': result.get('additional_metrics', {})
            }
            
        except Exception as e:
            print(f"实验失败: {algo_name} - {exp_config.name} - {str(e)}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - start_time,
                'instance_id': exp_config.name
            }
    
    def run_all_experiments(self, algorithm_names: List[str]) -> List[Dict[str, Any]]:
        """
        运行所有实验
        
        Args:
            algorithm_names: 要运行的算法名称列表
        """
        results = []
        total_experiments = len(self.experiment_configs) * len(self.seeds) * len(algorithm_names)
        completed = 0
        
        print(f"\n准备运行批量实验:")
        print(f"算法数量: {len(algorithm_names)}")
        print(f"算例数量: {len(self.experiment_configs)}")
        print(f"每个算例的种子数: {len(self.seeds)}")
        print(f"总实验数: {total_experiments}")
        print("="*60)
        
        # 显示算法列表
        print("要运行的算法:")
        for i, algo in enumerate(algorithm_names, 1):
            print(f"{i}. {algo}")
        print("="*60)
        
        for algo_name in algorithm_names:
            print(f"\n开始运行算法: {algo_name}")
            for config in self.experiment_configs:
                print(f"\n运行算例: {config.name}")
                for seed in self.seeds:
                    result = self.run_single_experiment(algo_name, config, seed)
                    results.append(result)
                    completed += 1
                    
                    # 显示进度
                    progress = (completed / total_experiments) * 100
                    print(f"总进度: {completed}/{total_experiments} ({progress:.1f}%)")
                    print("-" * 40)
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        total_time = sum(r.get('total_time', 0) for r in results)
        
        print("\n批量实验完成!")
        print(f"成功: {successful}/{total_experiments}")
        print(f"失败: {total_experiments - successful}/{total_experiments}")
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均每个实验耗时: {total_time/total_experiments:.2f}s")
        
        return results

def main():
    """主函数入口"""
    try:
        # 创建基础配置
        base_config = Config()
        
        # 定义要运行的算法
        algorithms = [
            "DQN",
            "PPO",
            "SARSA",
            "RuleDQN_DispatchHeuri",
            "Rule"
        ]
        
        # 创建实验运行器
        runner = MultiExperimentRunner(base_config)
        
        # 运行所有实验
        print("开始运行批量实验...")
        results = runner.run_all_experiments(algorithms)
        
        # 打印最终统计
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n实验结果摘要:")
        print(f"总实验数: {len(results)}")
        print(f"成功: {successful}")
        print(f"失败: {len(results) - successful}")
        print(f"结果已保存到: results/algorithm_comparison.csv")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
