"""
算例生成器：基于论文Table 2，生成FJSP-DP工序加工时间与交付要求的标准算例
"""
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from code.data.caseBuilder.config import Config

@dataclass
class FlexibleJobShopScenario:
    """FJSP-DP算例生成器
    
    基于配置生成包含加工时间和交付要求的标准算例。
    支持灵活的机器分配和批次调度需求。
    """
    
    def __init__(self, config: Config):
        """
        参数:
            config: 配置对象，包含所有必要参数
                - num_jobs: 工件数量
                - num_machines: 机器数量
                - num_distributors: 配送商数量
                - num_batches: 每个配送商的最大批次数
                - time_params: 时间相关参数
                - seed: 随机种子
        """
        self.config = config
        self.num_jobs = config.problem.num_jobs
        self.num_machines = config.problem.num_machines
        self.num_distributors = config.problem.num_distributors
        self.T_base = config.problem.batch_loading_base_time
        self.T_item = config.problem.item_loading_time
        self.seed = config.random_seed
        
        # 初始化数据结构
        self.job_num_ops: Dict[int, int] = {}
        self.available_machines: Dict[int, Dict[int, List[int]]] = {}
        self.processing_time: Dict[int, Dict[int, Dict[int, int]]] = {}
        self.job2distributor: Dict[int, int] = {}
        self.distributor2jobs: Dict[int, List[int]] = {}
        self.delivery_requirements: List[Dict] = []
        
        # 生成算例
        self._generate_case()
        
    def _generate_case(self):
        """生成完整的算例数据"""
        np.random.seed(self.seed)
        
        # 生成工序数量
        self._generate_operations()
        
        # 生成机器分配
        self._generate_machine_assignments()
        
        # 生成加工时间
        self._generate_processing_times()
        
        # 生成配送分配
        self._generate_distribution_assignments()
        
        # 生成交付要求
        self._generate_delivery_requirements()
        
    
    def _generate_operations(self):
        """为每个工件生成随机工序数"""
        min_ops = self.config.problem.min_operations
        max_ops = self.config.problem.max_operations
        self.job_num_ops = {
            j: np.random.randint(min_ops, max_ops + 1) 
            for j in range(self.num_jobs)
        }
    
    def _generate_machine_assignments(self):
        """生成工序-机器分配关系"""
        for j in range(self.num_jobs):
            self.available_machines[j] = {}
            for o in range(self.job_num_ops[j]):
                n_machines = np.random.randint(
                    self.config.problem.min_machines_per_op,
                    min(self.config.problem.max_machines_per_op, self.num_machines) + 1
                )
                self.available_machines[j][o] = sorted(
                    np.random.choice(
                        range(self.num_machines), 
                        n_machines, 
                        replace=False
                    ).tolist()
                )
    
    def _generate_processing_times(self):
        """生成加工时间"""
        for j in range(self.num_jobs):
            self.processing_time[j] = {}
            for o in range(self.job_num_ops[j]):
                self.processing_time[j][o] = {
                    m: int(np.random.randint(
                        self.config.problem.min_processing_time,
                        self.config.problem.max_processing_time
                    ))
                    for m in self.available_machines[j][o]
                }
    
    def _generate_distribution_assignments(self):
        """生成工件到配送商的分配"""
        # 分配工件到配送商
        self.job2distributor = {
            j: np.random.randint(0, self.num_distributors) 
            for j in range(self.num_jobs)
        }
        
        # 构建反向映射
        self.distributor2jobs = {
            d: [j for j, d2 in self.job2distributor.items() if d2 == d]
            for d in range(self.num_distributors)
        }
    
    def _generate_delivery_requirements(self):
        """生成交付要求"""
        min_time = self.config.problem.earliest_delivery_time
        max_time = self.config.problem.latest_delivery_time
        
        for d in range(self.num_distributors):
            jobs = self.distributor2jobs[d]
            if not jobs:
                continue
                
            n_req = np.random.randint(
                self.config.problem.min_delivery_requirements,
                self.config.problem.max_delivery_requirements + 1
            )
            
            ratios = np.linspace(1/n_req, 1, n_req)
            due_times = np.linspace(min_time, max_time, n_req).astype(int)
            
            for idx in range(n_req):
                weight = round(1.0 + idx/(n_req-1) if n_req > 1 else 1.0, 2)
                self.delivery_requirements.append({
                    "distributor_id": d,
                    "due_time": due_times[idx],
                    "ratio": round(ratios[idx], 2),
                    "weight": weight,
                    "jobs": jobs.copy()
                })
    
    
    def get_case_info(self) -> dict:
        """获取完整的算例信息"""
        return {
            "processing_times": self.get_processing_times_df(),
            "delivery_requirements": self.get_delivery_requirements_df(),
            "job_assignments": self.job2distributor,
            "available_machines": self.get_available_machines_df(),
            "loading_times": {
                "base": self.T_base,
                "per_item": self.T_item
            }
        }

    def get_processing_times_df(self) -> pd.DataFrame:
        """获取加工时间表"""
        records = []
        for job, ops in self.processing_time.items():
            for op, machines in ops.items():
                for machine, time in machines.items():
                    records.append({
                        "job_id": job,
                        "operation": op,
                        "machine": machine,
                        "processing_time": time
                    })

        # 向记录中添加工件对应的配送商
        for job in self.job2distributor:
            for record in records:
                if record["job_id"] == job:
                    record["distributor_id"] = self.job2distributor[job]
        return pd.DataFrame(records)
    


    def get_delivery_requirements_df(self) -> pd.DataFrame:
        """获取交付要求表"""
        return pd.DataFrame(self.delivery_requirements)
    

    def get_available_machines_df(self) -> pd.DataFrame:
        """获取可选机器表"""
        records = []
        for job, ops in self.available_machines.items():
            for op, machines in ops.items():
                records.append({
                    "job_id": job,
                    "operation": op,
                    "available_machines": machines
                })
        return pd.DataFrame(records)
    


    def summary(self):
        """打印算例摘要信息"""
        case_info = self.get_case_info()
        
       
        print("=== FJSP-DP算例信息 ===")
        print(f"\n基本配置:")
        print(f"工件数量: {self.num_jobs}")
        print(f"机器数量: {self.num_machines}")
        print(f"配送商数量: {self.num_distributors}")
            
        print("\n工序加工时间表：")
        print(case_info["processing_times"])
            
        print("\n交付要求表：")
        print(case_info["delivery_requirements"])
            
        print("\n可选机器表：")
        print(case_info["available_machines"])

        return case_info

import os

def save_caseAsCsv(case: FlexibleJobShopScenario, filename='../data/jobshopCase/case'):
    """将算例信息保存为CSV文件
    
    参数:
        case: JobShopCase实例
        filename: 保存文件的路径和前缀
    """
    case_info = case.get_case_info()
    job_num = case.num_jobs
    machines_num = case.num_machines
    distributors_num = case.num_distributors
    case_info["processing_times"].to_csv(f"{job_num}_{machines_num}_{distributors_num}_processing_times.csv", index=False)
    
    # 保存交付要求表
    case_info["delivery_requirements"].to_csv(f"{job_num}_{machines_num}_{distributors_num}_delivery_requirements.csv", 
        index=False
    )
    
    # 保存可选机器表
    case_info["available_machines"].to_csv(
        f"{job_num}_{machines_num}_{distributors_num}_available_machines.csv",
        index=False
    )

    print(f"算例信息已保存到:")


# 使用示例
if __name__ == "__main__":
    config = Config()
    case = FlexibleJobShopScenario(config)
    case_info = case.summary()
   # save_caseAsCsv(case)