class Config:
    train_mode = True
    
    # RL training parameters
    entropy_coef = 0.01  # Entropy regularization coefficient
    replay_buffer_size = 10000  # Experience replay buffer size
    batch_size = 64  # Training batch size
    gamma = 0.99  # Discount factor
    n_step = 3  # N-step TD learning
    ucb_exploration = 0.5  # UCB exploration coefficient
    
    num_jobs = 10
    num_machines = 2
    num_distributors = 2
    min_operations = 2
    max_operations = 2
    min_machines_per_op = 1
    max_machines_per_op = num_machines
    min_processing_time = 1
    max_processing_time = 4
    
    earliest_delivery_time = 1
    latest_delivery_time = 20
    min_delivery_requirements = 1
    max_delivery_requirements = 3
    min_load_ratio = 1


    max_time_steps = 100  # 增加时间步数让智能体有足够时间学习
    arrival_probability = 0
    max_job_num_limit = 20
    arrival_batch_size = 2
    num_machines = 5  # 增加机器数量缓解资源竞争

    urgent_threshold = 0
