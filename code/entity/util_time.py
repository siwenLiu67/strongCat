from dataclasses import dataclass, field

@dataclass
class TimeStamp:
    """时间戳管理"""
    value: float
    
    def time_until(self, target_time: float) -> float:
        """计算到目标时间的时间差"""
        return max(0, target_time - self.value)
    
    def is_valid_window(self, earliest: float, latest: float) -> bool:
        """检查时间窗口是否有效"""
        return earliest <= self.value <= latest