"""
实验结果分析脚本
Experiment Results Analysis Script
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 路径设置 ---
# 确保项目根目录在 sys.path 中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config

def analyze_results():
    """
    加载、分析并可视化实验结果。
    """
    results_csv_path = os.path.join(config.RESULTS_DIR, config.RESULTS_CSV_FILENAME)
    
    if not os.path.exists(results_csv_path):
        print(f"Error: Results file not found at '{results_csv_path}'.")
        print("Please run the experiment script first.")
        return

    print(f"--- Analyzing Results from '{results_csv_path}' ---")
    df = pd.read_csv(results_csv_path)

    # --- 数据清洗和预处理 ---
    # 转换 'instance' 为分类类型，以便更好地排序
    df['instance_type'] = df['instance'].apply(lambda x: x.split('_')[1] + '_' + x.split('_')[2])
    df['instance_type'] = pd.Categorical(df['instance_type'], 
                                         categories=sorted(df['instance_type'].unique()), 
                                         ordered=True)
    
    # --- 打印性能摘要 ---
    print("\n--- Performance Summary (Average Objective Value) ---")
    summary = df.groupby('algorithm')['objective_value'].mean().sort_values()
    print(summary.round(2).to_string())
    print("-" * 50)

    # --- 可视化 ---
    os.makedirs(config.ANALYSIS_FIGURES_DIR, exist_ok=True)
    
    # 1. 总体性能箱线图 (Boxplot of Overall Performance)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df, x='algorithm', y='objective_value', ax=ax)
    ax.set_title('Overall Performance Comparison (Objective Value)', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Objective Value (Lower is Better)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(config.ANALYSIS_FIGURES_DIR, 'overall_performance_boxplot.png')
    plt.savefig(fig_path)
    print(f"Saved overall performance boxplot to: {fig_path}")
    plt.close(fig)

    # 2. 按实例类型分组的性能条形图 (Bar Chart by Instance Type)
    fig, ax = plt.subplots(figsize=(15, 9))
    sns.barplot(data=df, x='instance_type', y='objective_value', hue='algorithm', ax=ax)
    ax.set_title('Performance by Instance Type', fontsize=16)
    ax.set_xlabel('Instance Type (Jobs x Machines)', fontsize=12)
    ax.set_ylabel('Average Objective Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = os.path.join(config.ANALYSIS_FIGURES_DIR, 'performance_by_instance_type.png')
    plt.savefig(fig_path)
    print(f"Saved performance by instance type bar chart to: {fig_path}")
    plt.close(fig)

    # 3. 执行时间比较 (Execution Time Comparison)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df, x='algorithm', y='execution_time_s', ax=ax, estimator=sum)
    ax.set_title('Total Execution Time Comparison', fontsize=16)
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Total Execution Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(config.ANALYSIS_FIGURES_DIR, 'execution_time_comparison.png')
    plt.savefig(fig_path)
    print(f"Saved execution time comparison to: {fig_path}")
    plt.close(fig)

    print("\n--- Analysis Finished ---")

if __name__ == "__main__":
    analyze_results()
