"""
算法基类
Base Class for All Algorithms
"""
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    所有求解算法的抽象基类。
    Abstract Base Class for all solving algorithms.

    强制所有子类实现一个 solve 方法。
    Enforces that all subclasses must implement a 'solve' method.
    """
    def __init__(self, env, config=None):
        """
        初始化基类。
        Initializes the base algorithm.

        :param env: FjspEnv, 调度环境实例。The scheduling environment instance.
        :param config: dict, 包含算法特定超参数的字典。A dictionary containing algorithm-specific hyperparameters.
        """
        self.env = env
        self.config = config if config is not None else {}
        self.n_jobs = env.num_jobs
        self.n_machines = env.num_machines
        self.n_ops = env.num_total_ops

    @abstractmethod
    def solve(self):
        """
        执行算法以找到问题的解。
        Executes the algorithm to find a solution to the problem.

        此方法必须在所有子类中被重写。
        This method MUST be overridden in all subclasses.

        :return: tuple (solution, results)
                 - solution: dict, 描述最终调度方案的字典。A dictionary describing the final schedule.
                 - results: dict, 包含评估指标的字典 (例如, 目标值, 完工时间等)。
                            A dictionary containing evaluation metrics (e.g., objective value, makespan, etc.).
        """
        pass

    def __repr__(self):
        """
        返回算法的官方字符串表示形式。
        Returns the official string representation of the algorithm.
        """
        return self.__class__.__name__
