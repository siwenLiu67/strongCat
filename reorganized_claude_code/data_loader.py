"""
Data loader for the reorganized Claude code.
Handles loading of instance data and configuration.
"""

import json
import os
from typing import Dict, Any, List
from .config import Config


def load_instance_data(instance_path: str) -> Dict[str, Any]:
    """
    Load instance data from JSON file.
    
    Args:
        instance_path: Path to the instance JSON file
        
    Returns:
        Dictionary containing instance data
    """
    with open(instance_path, 'r', encoding='utf-8') as f:
        instance_data = json.load(f)
    
    return instance_data


def load_job_shop_data(instance_path: str) -> Dict[str, Any]:
    """
    Load job shop specific data from instance file.
    
    Args:
        instance_path: Path to the instance JSON file
        
    Returns:
        Dictionary containing job shop data
    """
    instance_data = load_instance_data(instance_path)
    
    # Extract job shop data
    jobs = instance_data.get('jobs', {})
    machines = instance_data.get('machines', [])
    
    # Build precedence constraints
    precedence = {}
    for job_id, job_data in jobs.items():
        precedence[job_id] = {}
        operations = job_data.get('ops', [])
        for i in range(1, len(operations)):
            precedence[job_id][operations[i][0]] = [operations[i-1][0]]
    
    return {
        'jobs': jobs,
        'machines': machines,
        'precedence': precedence,
        'unit_production_cost': instance_data.get('meta', {}).get('unit_production_cost', 1.0)
    }


def load_dispatching_data(instance_path: str) -> Dict[str, Any]:
    """
    Load dispatching specific data from instance file.
    
    Args:
        instance_path: Path to the instance JSON file
        
    Returns:
        Dictionary containing dispatching data
    """
    instance_data = load_instance_data(instance_path)
    
    # Extract dispatching data
    transport_data = instance_data.get('transport_data', {})
    orders = instance_data.get('orders', {})
    
    # Extract markets from transport data
    markets = list(transport_data.keys())
    
    # Build job to market and job to deadline mappings
    job_to_market = {}
    job_to_deadline = {}
    for job_id, job_data in instance_data.get('jobs', {}).items():
        order_id = job_data.get('order_id')
        if order_id and order_id in orders:
            order_info = orders[order_id]
            job_to_market[job_id] = order_info.get('market')
            job_to_deadline[job_id] = order_info.get('due')
    
    return {
        'markets': markets,
        'transport_data': transport_data,
        'orders': orders,
        'job_to_market': job_to_market,
        'job_to_deadline': job_to_deadline
    }


def load_algorithm_parameters() -> Dict[str, Any]:
    """
    Load algorithm-specific parameters from configuration.
    
    Returns:
        Dictionary containing algorithm parameters
    """
    from .config import AlgorithmConfig
    
    return {
        'dqn': {
            'hidden_size': AlgorithmConfig.dqn_hidden_size,
            'target_update_freq': AlgorithmConfig.dqn_target_update_freq,
            'epsilon_start': AlgorithmConfig.dqn_epsilon_start,
            'epsilon_end': AlgorithmConfig.dqn_epsilon_end,
            'epsilon_decay': AlgorithmConfig.dqn_epsilon_decay
        },
        'ppo': {
            'clip_epsilon': AlgorithmConfig.ppo_clip_epsilon,
            'epochs': AlgorithmConfig.ppo_epochs,
            'value_coef': AlgorithmConfig.ppo_value_coef,
            'entropy_coef': AlgorithmConfig.ppo_entropy_coef
        },
        'sarsa': {
            'alpha': AlgorithmConfig.sarsa_alpha,
            'lambda': AlgorithmConfig.sarsa_lambda
        },
        'hiro': {
            'meta_learning_rate': AlgorithmConfig.hiro_meta_learning_rate,
            'intrinsic_reward_scale': AlgorithmConfig.hiro_intrinsic_reward_scale
        },
        'hrl_gnn': {
            'hidden_dim': AlgorithmConfig.hrl_gnn_hidden_dim,
            'num_layers': AlgorithmConfig.hrl_gnn_num_layers,
            'dropout': AlgorithmConfig.hrl_gnn_dropout
        },
        'optimal_critic': {
            'critic_lr': AlgorithmConfig.optimal_critic_critic_lr,
            'actor_lr': AlgorithmConfig.optimal_critic_actor_lr
        }
    }


def load_experiment_config() -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Returns:
        Dictionary containing experiment settings
    """
    from .config import ExperimentConfig
    
    return {
        'num_parallel_instances': ExperimentConfig.num_parallel_instances,
        'save_results': ExperimentConfig.save_results,
        'result_save_path': ExperimentConfig.result_save_path,
        'comparison_algorithms': ExperimentConfig.comparison_algorithms,
        'heuristic_algorithms': ExperimentConfig.heuristic_algorithms,
        'metrics': ExperimentConfig.metrics
    }


def get_available_instances(instances_dir: str = "instances") -> List[str]:
    """
    Get list of available instance files.
    
    Args:
        instances_dir: Directory containing instance files
        
    Returns:
        List of instance file paths
    """
    if not os.path.exists(instances_dir):
        return []
    
    instance_files = []
    for file in os.listdir(instances_dir):
        if file.endswith('.json'):
            instance_files.append(os.path.join(instances_dir, file))
    
    return sorted(instance_files)


if __name__ == '__main__':
    # Example usage
    instance_path = "instances/instance_m10_j20_s1.json"
    
    if os.path.exists(instance_path):
        # Load job shop data
        job_shop_data = load_job_shop_data(instance_path)
        print("Job Shop Data:")
        print(f"Jobs: {len(job_shop_data['jobs'])}")
        print(f"Machines: {len(job_shop_data['machines'])}")
        
        # Load dispatching data
        dispatching_data = load_dispatching_data(instance_path)
        print("\nDispatching Data:")
        print(f"Markets: {len(dispatching_data['markets'])}")
        print(f"Orders: {len(dispatching_data['orders'])}")
        
        # Load algorithm parameters
        algo_params = load_algorithm_parameters()
        print("\nAlgorithm Parameters:")
        print(f"DQN hidden size: {algo_params['dqn']['hidden_size']}")
        
        # Load experiment config
        exp_config = load_experiment_config()
        print(f"\nExperiment Config:")
        print(f"Comparison algorithms: {exp_config['comparison_algorithms']}")
        
        # Get available instances
        instances = get_available_instances()
        print(f"\nAvailable instances: {len(instances)}")
    else:
        print(f"Instance file {instance_path} not found")
