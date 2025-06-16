from config import Config
from typing import Dict
from data_structures import Job, Operation,Machine,Distributor
from case_generator import FlexibleJobShopScenario


class WarehouseEnvironment:
    """仓储-配送环境"""
    
    def __init__(self, config: Config, case: Dict):
        self.config = config
        self.case = case
        self.reset()


    def reset(self):
        self.t = 0
        self.done = False
     
               
        