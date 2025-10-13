"""
近端策略优化 (Proximal Policy Optimization, PPO) - 占位符
"""
from ..base_algorithm import BaseAlgorithm

class PPO_Agent(BaseAlgorithm):
    """
    PPO 算法的占位符实现。
    Placeholder implementation for the PPO algorithm.
    """
    def __init__(self, env, config):
        """
        :param env: FjspEnv, a scheduling environment instance.
        :param config: dict, a dictionary containing algorithm-specific hyperparameters.
        """
        super().__init__(env, config)
        self.is_implemented = self.config.get('is_implemented', False)

    def solve(self):
        """
        如果算法未实现，则打印消息并返回 None。
        """
        if not self.is_implemented:
            print(f"  {self.__class__.__name__} is not implemented. Skipping.")
            # 返回一个符合预期的空结果结构
            return None, {"objective_value": None, "is_feasible": False}
        
        # --- 未来的 PPO 实现将放在这里 ---
        # --- Future PPO implementation would go here ---
        
        print(f"Running {self.__class__.__name__}...")
        # ...
        print(f"{self.__class__.__name__} finished.")
        
        # 临时返回
        return None, {"objective_value": float('inf'), "is_feasible": False}
