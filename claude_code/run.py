from dataclasses import dataclass, field
from typing import List, Dict, Any
from config import Config
from case_generator import FlexibleJobShopScenario
from data_structures import Machine, Job, Operation, DeliveryRequirement

def main():
    """主函数"""
    # 1. 读取配置
    config = Config()

    # 2. 创建算例
    case = FlexibleJobShopScenario(config)
    # 打印算例信息
    print("生成的FJSP-DP算例:")
    print(f"作业数量: {len(case.jobs)}")
    print(f"机器数量: {len(case.machines)}")
    print(f"配送商数量: {len(case.distributors)}")

    # 3. 执行循环
    

if __name__ == "__main__":
    main()