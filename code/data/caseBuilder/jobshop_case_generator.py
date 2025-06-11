"""
算例生成器：基于论文Table 2，生成FJSP-DP工序加工时间与交付要求的标准算例
"""
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import sys
import os
# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from entity.config import Config
from entity.job_shop_entities import Job, Operation, Machine, DeliveryRequirement, DistributorAssignment

@dataclass
class FlexibleJobShopScenario:
    """FJSP-DP算例生成器
    
    使用实体类生成包含加工时间和交付要求的标准算例。
    支持灵活的机器分配和批次调度需求。
    """
    config: Config
    jobs: List[Job] = field(default_factory=list)
    machines: List[Machine] = field(default_factory=list)
    distributor_assignments: List[DistributorAssignment] = field(default_factory=list)
    delivery_requirements: List[DeliveryRequirement] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后自动生成算例"""
        self.num_jobs = self.config.problem.num_jobs
        self.num_machines = self.config.problem.num_machines
        self.num_distributors = self.config.problem.num_distributors
        self.T_base = self.config.problem.batch_loading_base_time
        self.T_item = self.config.problem.item_loading_time
        np.random.seed(self.config.random_seed)
        
        self._generate_case()
    
    def _generate_case(self):
        """生成完整的算例数据"""
        # 1. 初始化机器
        self._initialize_machines()
        
        # 2. 生成工件及其工序
        self._generate_jobs()
        
        # 3. 生成配送商分配
        self._generate_distributor_assignments()
        
        # 4. 生成交付要求
        self._generate_delivery_requirements()
    
    def _initialize_machines(self):
        """初始化机器列表"""
        self.machines = [
            Machine(
                machine_id=m,
                capabilities=[],
               # processing_speed=np.random.uniform(0.8, 1.2)  # 机器处理速度系数
            ) for m in range(self.num_machines)
        ]
    
    def _generate_jobs(self):
        """生成工件及其工序"""
        for j in range(self.num_jobs):
            # 生成工序数量
            num_ops = np.random.randint(
                self.config.problem.min_operations,
                self.config.problem.max_operations + 1
            )
            
            # 生成工序
            operations = []
            for o in range(num_ops):
                # 为工序分配机器
                n_machines = np.random.randint(
                    self.config.problem.min_machines_per_op,
                    min(self.config.problem.max_machines_per_op, self.num_machines) + 1
                )
                available_machines = sorted(
                    np.random.choice(range(self.num_machines), n_machines, replace=False)
                )
                
                # 生成加工时间
                processing_times = {
                    m: np.random.randint(
                        self.config.problem.min_processing_time,
                        self.config.problem.max_processing_time + 1
                    )
                    for m in available_machines
                }
                
                # 创建工序对象
                operation = Operation(
                    operation_id=o,
                    available_machines=available_machines,
                    processing_times=processing_times
                )
                operations.append(operation)
                
                # 更新机器能力
                for m_id in available_machines:
                    self.machines[m_id].capabilities.append(o)
            
            # 分配到配送商
            distributor_id = np.random.randint(0, self.num_distributors)
            
            # 创建工件对象
            job = Job(
                job_id=j,
                operations=operations,
                distributor_id=distributor_id
              #  priority=np.random.randint(1, self.config.problem.max_priority + 1)
            )
            self.jobs.append(job)
    
    def _generate_distributor_assignments(self):
        """生成配送商分配"""
        for d in range(self.num_distributors):
            assigned_jobs = [j.job_id for j in self.jobs if j.distributor_id == d]
            assignment = DistributorAssignment(
                distributor_id=d,
                assigned_jobs=assigned_jobs
            )
            self.distributor_assignments.append(assignment)
    
    def _generate_delivery_requirements(self):
        """生成交付要求"""
        min_time = self.config.problem.earliest_delivery_time
        max_time = self.config.problem.latest_delivery_time
        
        for dist in self.distributor_assignments:
            if not dist.assigned_jobs:
                continue
                
            n_req = np.random.randint(
                self.config.problem.min_delivery_requirements,
                self.config.problem.max_delivery_requirements + 1
            )
            
            ratios = np.linspace(1/n_req, 1, n_req)
            due_times = np.linspace(min_time, max_time, n_req).astype(int)
            
            for idx in range(n_req):
                weight = round(1.0 + idx/(n_req-1) if n_req > 1 else 1.0, 2)
                requirement = DeliveryRequirement(
                    distributor_id=dist.distributor_id,
                    due_time=int(due_times[idx]),
                    ratio=round(ratios[idx], 2),
                    weight=weight,
                    jobs=dist.assigned_jobs.copy()
                )
                self.delivery_requirements.append(requirement)
                dist.delivery_requirements.append(requirement)
    
    def get_case_info(self) -> dict:
        """获取完整的算例信息"""
        return {
            "processing_times": self.get_processing_times_df(),
            "delivery_requirements": self.get_delivery_requirements_df(),
            "available_machines": self.get_available_machines_df(),
            "loading_times": {
                "base": self.T_base,
                "per_item": self.T_item
            }
        }
    
    def get_processing_times_df(self) -> pd.DataFrame:
        """获取加工时间表"""
        records = []
        for job in self.jobs:
            for op in job.operations:
                for machine, time in op.processing_times.items():
                    records.append({
                        "job_id": job.job_id,
                        "operation": op.operation_id,
                        "machine": machine,
                        "processing_time": time,
                        "distributor_id": job.distributor_id
                    })
        return pd.DataFrame(records)
    
    def get_delivery_requirements_df(self) -> pd.DataFrame:
        """获取交付要求表"""
        records = [
            {
                "distributor_id": req.distributor_id,
                "due_time": req.due_time,
                "ratio": req.ratio,
                "weight": req.weight,
                "jobs": req.jobs
            }
            for req in self.delivery_requirements
        ]
        return pd.DataFrame(records)
    
    def get_available_machines_df(self) -> pd.DataFrame:
        """获取可选机器表"""
        records = []
        for job in self.jobs:
            for op in job.operations:
                records.append({
                    "job_id": job.job_id,
                    "operation": op.operation_id,
                    "available_machines": op.available_machines
                })
        return pd.DataFrame(records)
    
    def summary(self) -> dict:
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