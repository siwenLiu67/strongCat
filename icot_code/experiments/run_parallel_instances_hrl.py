import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from icot_code.data_loader import load_production_data, load_transportation_data, load_drl_hyperparameters, load_search_parameters
from icot_code.models.experiment_result import ExperimentResult, save_experiment_result
from icot_code.experiments.main import define_search_space, evaluate_t_internal

def run_single_instance_parallel(file_path, instance_id):
    # 复制 main_single_instance 的主要流程
    prod_data = load_production_data(file_path)
    trans_data = load_transportation_data(file_path)
    drl_params = load_drl_hyperparameters()
    search_params = load_search_parameters()
    prod = prod_data
    print(f"[{instance_id}] Production summary:")
    total_ops = 0
    min_proc = float('inf')
    max_proc = 0
    sum_min_proc = 0
    for jid, ops in prod['jobs'].items():
        total_ops += len(ops)
        for op_id, m_times in ops:
            if m_times:
                mm = min(m_times.values())
                min_proc = min(min_proc, mm)
                max_proc = max(max_proc, mm)
                sum_min_proc += mm
    print(f"[{instance_id}] jobs: {len(prod['jobs'])}, total ops: {total_ops}")
    print(f"[{instance_id}] min op time: {min_proc}, max op time: {max_proc}, sum of min times (LB): {sum_min_proc}")
    import time
    start_time = time.time()
    lower_bound, upper_bound = define_search_space(prod_data, trans_data)
    print(f"[{instance_id}] T_internal 搜索空间定义为: [{lower_bound}, {upper_bound}]")
    import numpy as np
    coarse_search_space = np.arange(lower_bound, upper_bound, search_params['coarse_step'])
    results = []
    best_rewards_curve = []  # 新增：记录每次粗略/精细搜索的reward曲线
    for t_internal in coarse_search_space:
        result = evaluate_t_internal(
            t_internal,
            prod_data=prod_data,
            trans_data=trans_data,
            drl_params=drl_params,
            max_episodes=drl_params['coarse_search_max_episodes']
        )
        results.append(result)
    best_total_cost = float('inf')
    best_t_internal = -1
    best_plan = None
    for t_internal, total_cost, plan, c_max in results:
        if total_cost < best_total_cost:
            best_total_cost = total_cost
            best_t_internal = t_internal
            best_plan = plan
    print(f"[{instance_id}] --- 粗略搜索完成 ---")
    print(f"[{instance_id}] 最佳 T_internal (粗略): {best_t_internal}, 成本: {best_total_cost}")
    if best_t_internal != -1:
        print(f"[{instance_id}] --- 开始精细网格搜索 ---")
        fine_search_lower = max(lower_bound, best_t_internal - search_params['fine_search_range'] // 2)
        fine_search_upper = min(upper_bound, best_t_internal + search_params['fine_search_range'] // 2)
        fine_search_space = np.arange(fine_search_lower, fine_search_upper, search_params['fine_step'])
        fine_results = []
        for t_internal in fine_search_space:
            if t_internal in [res[0] for res in results]:
                continue
            result = evaluate_t_internal(
                t_internal,
                prod_data=prod_data,
                trans_data=trans_data,
                drl_params=drl_params,
                max_episodes=drl_params['max_episodes']
            )
        
            fine_results.append(result)
        results.extend(fine_results)
        results.sort(key=lambda x: x[0])
        for t_internal, total_cost, plan, c_max in results:
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_t_internal = t_internal
                best_plan = plan
                print(f"[{instance_id}] *** 找到新的最佳解决方案! T_internal={best_t_internal}, 成本={best_total_cost} ***")
    endtime = time.time()
    computation_time = endtime - start_time
    print(f"[{instance_id}] --- HD-DRL 算法完成 ---")
    if best_t_internal != -1 and best_plan and best_plan.get('production'):
        c_max_final = max(op['end'] for op in best_plan['production'])
    else:
        c_max_final = float('inf')
    metrics = {
        "total_cost": best_total_cost,
        "production_cost": c_max_final * prod_data['c_unit_production'] if best_plan and best_plan.get('production') else float('inf'),
        "transportation_cost": best_total_cost - c_max_final * prod_data['c_unit_production'],
        "computation_time": computation_time,
    }
    result = ExperimentResult(
        algorithm_name="hrl_ppo",
        instance_id=instance_id,
        metrics=metrics,
        extra_info={"reward_curve": best_plan.get('reward_curve', [])}  # 新增：保存reward曲线  
    )
    save_experiment_result(result, save_dir="results", filetype="csv")
    print(f"[{instance_id}] 结果已保存。")
    return instance_id, best_total_cost

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
            futures.append(executor.submit(run_single_instance_parallel, file_path, instance_id))
        for future in as_completed(futures):
            try:
                instance_id, cost = future.result()
                print(f"[{instance_id}] 运行完成，总成本: {cost}")
            except Exception as e:
                print(f"某个实例运行出错: {e}")

if __name__ == '__main__':
    main()