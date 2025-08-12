class Config:
    train_mode = True
    episodes =  2
    seeds = [42]
    
    # RL training parameters
    entropy_coef = 0.01  # Entropy regularization coefficient
    replay_buffer_size = 2000  # Experience replay buffer size
    batch_size = 64  # Training batch size
    gamma = 0.95  # Discount factor
    n_step = 3  # N-step TD learning
    ucb_exploration = 0.5  # UCB exploration coefficient
    learning_rate = 0.001

    num_initial_jobs = 4
    num_dynamic_jobs = 10
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
    earliest_arrival_time: int = 10  # 最早到达时间
   
    # 批量到达参数（可选）
    batch_arrival_probability: float = 1  # 批量到达概率
    max_batch_size: int = 10  # 最大批量到达数量

    earliest_delivery_time = 1
    latest_delivery_time = 20
    min_delivery_requirements = 1
    max_delivery_requirements = 3
    min_load_ratio = 1

    # Due date相关配置
    dispatch_preparation_time: int = 5  # 配送准备时间
    min_due_date_factor: float = 1.1    # 最小due date系数
    
    # 重要性计算权重
    amount_weight: float = 0.6          # 数量权重
    complexity_weight: float = 0.4      # 复杂度权重

    max_time_steps = 300  # 增加时间步数让智能体有足够时间学习
    arrival_probability = 0
    max_job_num_limit = 20
    arrival_batch_size = 2
    


    urgent_threshold = 0
