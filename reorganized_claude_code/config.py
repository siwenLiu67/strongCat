"""
Configuration management for the reorganized Claude code.
Follows modular architecture similar to icot_code.
"""

class Config:
    """Main configuration class for the integrated scheduling and dispatching system."""
    
    # ===== General Settings =====
    train_mode = True
    episodes = 2
    seeds = [42]
    max_time_steps = 200
    
    # ===== RL Training Parameters =====
    entropy_coef = 0.01  # Entropy regularization coefficient
    replay_buffer_size = 2000  # Experience replay buffer size
    batch_size = 64  # Training batch size
    gamma = 0.95  # Discount factor
    n_step = 3  # N-step TD learning
    ucb_exploration = 0.5  # UCB exploration coefficient
    learning_rate = 0.001

    # ===== Job Shop Parameters =====
    num_initial_jobs = 4
    num_dynamic_jobs = 10
    num_machines = 2
    min_operations = 1
    max_operations = 20
    min_machines_per_op = 1
    max_machines_per_op = 5
    min_processing_time = 1
    max_processing_time = 1

    # ===== Job Arrival Parameters =====
    arrival_probability = 0
    max_job_num_limit = 20
    arrival_batch_size = 2
    earliest_arrival_time = 10  # 最早到达时间
    
    # Batch arrival parameters (optional)
    batch_arrival_probability = 0  # 批量到达概率
    max_batch_size = 10  # 最大批量到达数量

    # ===== Job Amount Parameters =====
    max_job_amount = 100
    min_job_amount = 50

    # ===== Dispatching Parameters =====
    num_distributors = 2
    earliest_delivery_time = 1
    latest_delivery_time = 20
    min_delivery_requirements = 1
    max_delivery_requirements = 3
    min_load_ratio = 0.0

    # ===== Due Date Parameters =====
    dispatch_preparation_time = 5  # 配送准备时间
    min_due_date_factor = 1.1    # 最小due date系数
    
    # ===== Importance Calculation Weights =====
    amount_weight = 0.6          # 数量权重
    complexity_weight = 0.4      # 复杂度权重

    # ===== CP-SAT Parameters =====
    big_m = 10000  # 大M值
    solver_time_limit = 30  # 求解器时间限制（秒）
    urgent_threshold = 0


class ExperimentConfig:
    """Configuration for experiment settings."""
    
    # Batch experiment settings
    num_parallel_instances = 5
    save_results = True
    result_save_path = "results/"
    
    # Algorithm comparison settings
    comparison_algorithms = ["DQN", "PPO", "SARSA", "HIRO", "HRL-GNN", "OptimalCritic"]
    heuristic_algorithms = ["DispatchHeuristic", "DynamicPriorityRule"]
    
    # Performance metrics
    metrics = ["makespan", "total_cost", "tardiness", "utilization"]


class AlgorithmConfig:
    """Configuration for algorithm-specific parameters."""
    
    # DQN parameters
    dqn_hidden_size = 128
    dqn_target_update_freq = 100
    dqn_epsilon_start = 1.0
    dqn_epsilon_end = 0.01
    dqn_epsilon_decay = 0.995
    
    # PPO parameters
    ppo_clip_epsilon = 0.2
    ppo_epochs = 4
    ppo_value_coef = 0.5
    ppo_entropy_coef = 0.01
    
    # SARSA parameters
    sarsa_alpha = 0.1
    sarsa_lambda = 0.9
    
    # HIRO parameters
    hiro_meta_learning_rate = 0.0001
    hiro_intrinsic_reward_scale = 0.1
    
    # HRL-GNN parameters
    hrl_gnn_hidden_dim = 64
    hrl_gnn_num_layers = 3
    hrl_gnn_dropout = 0.1
    
    # Optimal Critic parameters
    optimal_critic_critic_lr = 0.001
    optimal_critic_actor_lr = 0.0001
