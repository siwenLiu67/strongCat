class ExperimentResult:
    def __init__(
        self,
        algorithm_name: str,
        instance_id: str,
        metrics: dict,
        extra_info: dict = None
    ):
        self.algorithm_name = algorithm_name
        self.instance_id = instance_id   
        self.metrics = metrics
        self.extra_info = extra_info or {}

    def to_dict(self):
        return {
            "algorithm_name": self.algorithm_name,
            "instance_id": self.instance_id,
            "metrics": self.metrics,
            "extra_info": self.extra_info
        }
    
    import json
import pandas as pd
import os

def save_experiment_result(result: ExperimentResult, save_dir: str, filetype: str = "csv"):
    """
    所有结果都追加保存到一个文件（如 all_experiment_results.csv）中
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"all_experiment_results.{filetype}")
    
    df = pd.DataFrame([result.to_dict()])
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
    print(f"结果已保存到 {filename}")