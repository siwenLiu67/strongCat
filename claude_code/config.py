class Config:
    train_mode = True
    episodes = 1  # 增加训练轮次，提供更多收敛机会
    seeds = [42]

    meta_decision_interval = 10  # 高层决策间隔（时间步数）
    
    # RL training parameters - 收敛优化配置
    entropy_coef = 0.001  # 减少熵正则化，避免过度探索
    replay_buffer_size = 10000  # 减少缓冲区大小，提高学习效率
    batch_size = 128  # 减少批量大小，提高训练稳定性
    gamma = 0.95  # 降低折扣因子，关注近期奖励
    n_step = 3  # 减少TD学习步数，降低方差
    ucb_exploration = 0.02  # 减少探索系数，提高利用
    learning_rate = 0.0003  # 稍微提高学习率，加快收敛
    
    # 收敛优化参数
    target_update_frequency = 30  # 减少目标网络更新频率
    gradient_clip = 0.3  # 减少梯度裁剪，提高稳定性
    epsilon_start = 0.8  # 降低初始探索率
    epsilon_end = 0.05  # 提高最终探索率，保持一定探索
    epsilon_decay = 0.98  # 减缓探索率衰减
    
    # 收敛增强参数
    warmup_episodes = 15  # 减少预热阶段
    learning_rate_decay = 0.99  # 减缓学习率衰减
    min_learning_rate = 1e-5  # 提高最小学习率
    reward_clip = 8.0  # 减少奖励裁剪阈值
    
    # 高级收敛参数
    adaptive_lr = True  # 自适应学习率
    patience_threshold = 15  # 减少早停耐心阈值
    convergence_window = 20  # 减少收敛检测窗口大小
    min_improvement = 0.01  # 降低最小改进阈值
    
    # DQN相关配置
    epsilon_start = 0.8
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update_frequency = 100

    num_initial_jobs = 4
    num_dynamic_jobs = 5  # 减少动态作业数量，确保能到达
    num_machines = 2
    num_distributors = 2
    min_operations = 1
    max_operations = 20
    min_machines_per_op = 1
    max_machines_per_op = 5
    min_processing_time = 1
    max_processing_time = 1
    max_job_amount = 100
    min_job_amount = 50
    earliest_arrival_time: int = 5  # 更早的到达时间
   
    # 批量到达参数（修复动态作业到达）
    batch_arrival_probability: float = 0.8  # 提高批量到达概率
    max_batch_size: int = 3  # 减少批量大小，确保能到达
    min_batch_size: int = 1  # 添加最小批量大小

    earliest_delivery_time = 1
    latest_delivery_time = 20
    min_delivery_requirements = 1
    max_delivery_requirements = 3
    min_load_ratio = 0.0

    # Due date相关配置
    dispatch_preparation_time: int = 5  # 配送准备时间
    min_due_date_factor: float = 1.1    # 最小due date系数
    
    # 重要性计算权重
    amount_weight: float = 0.6          # 数量权重
    complexity_weight: float = 0.4      # 复杂度权重

    max_time_steps = 200  # 增加时间步数让智能体有足够时间学习
    arrival_probability = 0
    max_job_num_limit = 20
    arrival_batch_size = 2
    


    # CP-SAT相关参数
    big_m = 10000  # 大M值
    solver_time_limit = 30  # 求解器时间限制（秒）
    urgent_threshold = 0
