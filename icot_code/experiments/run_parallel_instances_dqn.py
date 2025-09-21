import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from icot_code.data_loader import load_production_data, load_transportation_data
from icot_code.models.experiment_result import ExperimentResult, save_experiment_result
from icot_code.comparison_algorithms.single_layer_drl import DQNAlgorithm

def run_single_instance_dqn(file_path, instance_id):
    production_data = load_production_data(file_path)
    transportation_data = load_transportation_data(file_path)
    orders_data = transportation_data['orders']
    print(f"[{instance_id}] 开始运行 DQN 算法...")
    dqn = DQNAlgorithm()
    result = dqn.solve(production_data, transportation_data, orders_data)
    print(f"[{instance_id}] DQN 调度完成 (总成本={result['metrics']['total_cost']})")
    experiment_result = ExperimentResult(
        algorithm_name=dqn.name,
        instance_id=instance_id,
        metrics=result['metrics'],
        extra_info={'reward_curve': result.get('extra_info', {}).get('reward_curve', [])}
    )
    save_experiment_result(experiment_result, save_dir="results", filetype="csv")
    print(f"[{instance_id}] 结果已保存。")
    return instance_id, result['metrics']['total_cost']

def main():
    # 1. 遍历所有实例，按m和n分组
    instances_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'instances')
    all_files = [f for f in os.listdir(instances_dir) if f.endswith('.json')]
    group_dict = {}
    for fname in all_files:
        match = re.search(r'm(\d+)_j(\d+)', fname)
        if match:
            m = int(match.group(1))
            n = int(match.group(2))
            group_dict.setdefault((m, n), []).append(fname)
    print(f"共发现{len(group_dict)}组(m, n)实例")
    # 2. 用户输入
    m_input = int(input("请输入要运行的机器数m: "))
    n_input = int(input("请输入要运行的工件数n: "))
    key = (m_input, n_input)
    if key not in group_dict:
        print(f"未找到m={m_input}, n={n_input}的实例组！")
        return
    files = group_dict[key]
    print(f"将并行运行分组: m={m_input}, n={n_input}, 共{len(files)}个实例")
    # 3. 并行运行
    with ProcessPoolExecutor(max_workers=min(3, len(files))) as executor:
        futures = []
        for fname in files:
            instance_id = fname.replace('.json', '')
            file_path = os.path.join(instances_dir, fname)
            futures.append(executor.submit(run_single_instance_dqn, file_path, instance_id))
        for future in as_completed(futures):
            try:
                instance_id, cost = future.result()
                print(f"[{instance_id}] 运行完成，总成本: {cost}")
            except Exception as e:
                print(f"某个实例运行出错: {e}")

if __name__ == '__main__':
    main()
