import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import os

def generate_comparison_plots(test_results):
    """Generate comparison plots of different metrics"""
    metrics = ['makespan', 'tardiness', 'utilization']
    labels = ['Makespan (hours)', 'Tardiness (hours)', 'Utilization (%)']
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics):
        values = [res[metric] for res in test_results]
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(values)), values)
        plt.title(f'Test {metric.capitalize()}')
        plt.xlabel('Test Scenario')
        plt.ylabel(labels[i])
    plt.tight_layout()
    plt.savefig('results/test_comparison.png')
    plt.close()

def plot_gantt_chart(schedule, filename='results/gantt_chart.png'):
    """Generate Gantt chart visualization of schedule"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = cm.get_cmap('tab20', len(schedule['jobs']))
    
    for i, job in enumerate(schedule['jobs']):
        for op in job['operations']:
            ax.barh(
                y=op['machine'],
                width=op['end']-op['start'],
                left=op['start'],
                color=colors(i),
                edgecolor='black',
                label=f'Job {job["id"]}'
            )
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Job Shop Schedule')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_training_metrics(metrics, filename='results/training_metrics.json'):
    """Save training metrics to JSON file"""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_training_metrics(filename='results/training_metrics.json'):
    """Load training metrics from JSON file"""
    if os.path.exists(filename):
        with open(filename) as f:
            return json.load(f)
    return None

def visualize_attention_weights(weights, filename='results/attention_weights.png'):
    """Visualize attention weights from Transformer"""
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Input Sequence')
    plt.ylabel('Output Sequence')
    plt.savefig(filename)
    plt.close()

def generate_performance_report(test_results):
    """Generate comprehensive performance report"""
    report = {
        'average_makespan': np.mean([res['makespan'] for res in test_results]),
        'average_tardiness': np.mean([res['tardiness'] for res in test_results]),
        'average_utilization': np.mean([res['utilization'] for res in test_results]),
        'best_scenario': min(test_results, key=lambda x: x['makespan']),
        'worst_scenario': max(test_results, key=lambda x: x['makespan'])
    }
    
    with open('results/performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report
