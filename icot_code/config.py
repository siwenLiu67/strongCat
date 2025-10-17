# config.py

# DRL (深度强化学习) 超参数
GAMMA = 0.95  # 折扣因子
LEARNING_RATE = 0.001  # 学习率
MAX_EPISODES = 50  # 精细搜索或独立运行时的最大训练轮次
COARSE_SEARCH_MAX_EPISODES = 50  # 自动调参
K = 1000  # 用于惩罚的大的正常数
ALPHA = 1.5  # 超出 T_internal 的惩罚系数
BETA = 1.0  # 在 T_internal 内完成的奖励系数

# T_internal (内部生产截止时间) 的搜索空间参数
T_INTERNAL_COARSE_STEP = 10  # 自动调参
T_INTERNAL_FINE_STEP = 3  # 自动调参
T_INTERNAL_FINE_SEARCH_RANGE = 50  # 自动调参

# 生产参数
C_UNIT_PRODUCTION = 10  # 单位时间生产成本
M_BIG = 10000  # 用于混合整数线性规划(MILP)约束的大数 (在DRL中未使用，但作为良好实践保留)

# 调度规则 (用于 DRL 动作空间)
# 可用规则: 'SPT', 'LPT', 'FIFO', 'EDD', 'CR'
SCHEDULING_RULES = ['SPT', 'LPT', 'FIFO', 'EDD', 'CR']


# --- 示例问题实例数据 ---
# 在实际应用中，这些数据会从一个单独的文件加载 (例如, data_loader.py)

# 机器集合
MACHINES = ['M1', 'M2', 'M3']

# 工件集合与工序
# 结构: {工件ID: [(工序ID, {机器ID: 加工时间, ...}), ...]}
JOBS = {
    'J1': [
        ('O11', {'M1': 5, 'M2': 7}),
        ('O12', {'M2': 6, 'M3': 8})
    ],
    'J2': [
        ('O21', {'M1': 8, 'M2': 10, 'M3': 4}),
        ('O22', {'M1': 5, 'M3': 6})
    ]
}
# 紧前工序约束: Pred(j,o)
# 结构: {工件ID: {工序ID: [紧前工序ID列表]}}
PRECEDENCE = {
    'J1': {'O12': ['O11']},
    'J2': {'O22': ['O21']}
}


# 市场集合
MARKETS = ['US', 'Australia', 'Germany', 'Japan', 'France', 'Netherlands']

# 运输参数
# 结构: {市场: {运输方式: {服务商: {参数: 值}}}}
TRANSPORT_DATA = {
    'US': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 19446},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 19446},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 19446}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 10400},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 10400},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 10400}
        }
    },
    'Australia': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 11112},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 11112},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 11112}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 7800},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 7800},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 7800}
        }
    },
    'Germany': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 20000},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 20000},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 20000}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 9200},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 9200},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 9200}
        },
        'Land': {
            'P1': {'cost': 0.086, 'speed': 80, 'distance': 11000},
            'P2': {'cost': 0.095, 'speed': 90, 'distance': 11000},
            'P3': {'cost': 0.105, 'speed': 100, 'distance': 11000}
        }
    },
    'Japan': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 3148},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 3148},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 3148}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 1200},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 1200},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 1200}
        }
    },
    'France': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 20378},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 20378},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 20378}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 12500},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 12500},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 12500}
        },
        'Land': {
            'P1': {'cost': 0.086, 'speed': 80, 'distance': 12500},
            'P2': {'cost': 0.095, 'speed': 90, 'distance': 12500},
            'P3': {'cost': 0.105, 'speed': 100, 'distance': 12500}
        }
    },
    'Netherlands': {
        'Sea': {
            'P1': {'cost': 0.11, 'speed': 25, 'distance': 19446},
            'P2': {'cost': 0.09, 'speed': 20, 'distance': 19446},
            'P3': {'cost': 0.13, 'speed': 28, 'distance': 19446}
        },
        'Air': {
            'P1': {'cost': 2.5, 'speed': 850, 'distance': 9000},
            'P2': {'cost': 3.2, 'speed': 820, 'distance': 9000},
            'P3': {'cost': 1.8, 'speed': 800, 'distance': 9000}
        },
        'Land': {
            'P1': {'cost': 0.086, 'speed': 80, 'distance': 12000},
            'P2': {'cost': 0.095, 'speed': 90, 'distance': 12000},
            'P3': {'cost': 0.105, 'speed': 100, 'distance': 12000}
        }
    }
}

# 订单详情
# 结构: {市场: {'weight': W_d, 'deadline': T_deadline_d}}
ORDERS = {
    'US': {'weight': 10, 'deadline': 600},
    'Australia': {'weight': 7, 'deadline': 700},
    'Germany': {'weight': 8, 'deadline': 500}, # Placeholder
    'Japan': {'weight': 5, 'deadline': 550},
    'France': {'weight': 8, 'deadline': 500}, # Placeholder
    'Netherlands': {'weight': 8, 'deadline': 500} # Placeholder
}
