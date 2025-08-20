#!/usr/bin/env python3
"""
测试脚本用于验证environment.py的日志精简功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'claude_code'))

from environment import WarehouseEnvironment
from config import Config

def test_logging_reduction():
    print("=== 测试日志精简功能 ===")
    
    # 测试训练模式下的日志输出
    print("\n1. 训练模式 (train_mode=True):")
    config = Config()
    config.train_mode = True
    env = WarehouseEnvironment(config)
    
    # 重置环境，应该看到日志输出
    print("\n重置环境:")
    env.reset()
    
    # 测试非训练模式下的日志输出
    print("\n2. 非训练模式 (train_mode=False):")
    config.train_mode = False
    env = WarehouseEnvironment(config)
    
    # 重置环境，应该看不到日志输出
    print("\n重置环境 (应该无日志输出):")
    env.reset()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_logging_reduction()
