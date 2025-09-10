# data_loader.py

import icot_code.config as config
# def load_production_data():
#     """
#     从配置文件加载所有与生产相关的数据。
#     """
#     return {
#         'jobs': config.JOBS,
#         'machines': config.MACHINES,
#         'precedence': config.PRECEDENCE,
#         'c_unit_production': config.C_UNIT_PRODUCTION
#     }

# def load_transportation_data():
#     """
#     从配置文件加载所有与运输相关的数据。
#     """
#     return {
#         'markets': config.MARKETS,
#         'transport_data': config.TRANSPORT_DATA,
#         'orders': config.ORDERS
#     }



import json

def load_production_data(file_path):
    """
    从JSON配置文件加载所有与生产相关的数据。

    Args:
        file_path (str): JSON文件路径

    Returns:
        dict: 包含生产相关数据的字典，格式为:
        {
            'jobs': dict,       # 工件数据
            'machines': list,   # 机器列表
            'precedence': dict, # 紧前工序约束
            'c_unit_production': float  # 单位生产成本
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 转换jobs格式以匹配PRECEDENCE结构
    jobs = {}
    for job_id, job_data in config['jobs'].items():
        jobs[job_id] = job_data['ops']
    
    # 构造 job_id -> market 和 job_id -> deadline 的映射
    job_to_market = {}
    job_to_deadline = {}
    orders_data = config['orders']
    for job_id, job_details in config['jobs'].items():
        order_id = job_details.get('order_id')
        if order_id and order_id in orders_data:
            order_info = orders_data[order_id]
            job_to_market[job_id] = order_info.get('market')
            job_to_deadline[job_id] = order_info.get('due')

    
    # 构建precedence约束（根据示例数据结构推断）
    precedence = {}
    for job_id, job_data in config['jobs'].items():
        precedence[job_id] = {}
        ops = job_data['ops']
        for i in range(1, len(ops)):
            precedence[job_id][ops[i][0]] = [ops[i-1][0]]
    
    return {
        'jobs': jobs,
        'machines': config['machines'],
        'precedence': precedence,
        'c_unit_production': config['meta']['unit_production_cost'],
        'job_to_market': job_to_market,
        'job_to_deadline': job_to_deadline
    }

def load_transportation_data(file_path):
    """
    从JSON配置文件加载所有与运输相关的数据。

    Args:
        file_path (str): JSON文件路径

    Returns:
        dict: 包含运输相关数据的字典，格式为:
        {
            'markets': list,           # 市场列表
            'transport_data': dict,    # 运输参数数据
            'orders': dict             # 订单数据
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 提取所有唯一市场
    markets = list(config['transport_data'].keys())
    
    return {
        'markets': markets,
        'transport_data': config['transport_data'],
        'orders': config['orders']
    }




def load_drl_hyperparameters():
    """
    从配置文件加载所有与DRL相关的超参数。
    """
    return {
        'gamma': config.GAMMA,
        'learning_rate': config.LEARNING_RATE,
        'max_episodes': config.MAX_EPISODES,
        'coarse_search_max_episodes': config.COARSE_SEARCH_MAX_EPISODES,
        'K': config.K,
        'ALPHA': config.ALPHA,
        'BETA': config.BETA
    }

def load_search_parameters():
    """
    加载用于T_internal优化的搜索参数。
    """
    return {
        'coarse_step': config.T_INTERNAL_COARSE_STEP,
        'fine_step': config.T_INTERNAL_FINE_STEP,
        'fine_search_range': config.T_INTERNAL_FINE_SEARCH_RANGE
    }

if __name__ == '__main__':
    # 如何使用加载函数的示例
    file_path = "./instances/instance_m10_j20_s1.json"
    
    # 加载生产数据
    production_data = load_production_data(file_path)
    print("生产数据加载完成:")
    print(f"机器数量: {len(production_data['machines'])}")
    print(f"工件数量: {len(production_data['jobs'])}")
    print(f"单位生产成本: {production_data['c_unit_production']}")
    
    # 加载运输数据
    transportation_data = load_transportation_data(file_path)
    print("\n运输数据加载完成:")
    print(f"市场数量: {len(transportation_data['markets'])}")
    print(f"订单数量: {len(transportation_data['orders'])}")
    print(f"运输方式: {list(transportation_data['transport_data']['US'].keys())}")


    drl_params = load_drl_hyperparameters()
    search_params = load_search_parameters()

    print("--- 生产数据 ---")
    print(production_data)
    print("\n--- 运输数据 ---")
    print(transportation_data)
    print("\n--- DRL 超参数 ---")
    print(drl_params)
    print("\n--- 搜索参数 ---")
    print(search_params)