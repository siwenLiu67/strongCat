"""
生成结构化的 FJSP 实例 JSON 文件。
"""
import json
import os
import random

def generate_fjsp_instance(num_jobs, num_machines, ops_per_job_range, proc_time_range, num_transporters):
    """
    生成一个与 FjspEnv 期望的结构完全匹配的 FJSP 实例。
    """
    # --- 基础结构 ---
    instance = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "num_transporters": num_transporters,
        "transporters": [],
        "constraints": {},
        "weights": {
            "makespan": 0.4,
            "energy": 0.3,
            "carbon": 0.3
        },
        "jobs": []
    }

    # --- 生成运输工具 ---
    for t_id in range(num_transporters):
        instance["transporters"].append({
            "transporter_id": t_id,
            "transport_time": random.randint(5, 20),
            "carbon_factor": round(random.uniform(0.1, 0.5), 2)
        })

    total_proc_time = 0
    
    # --- 生成作业和工序 ---
    for j_id in range(num_jobs):
        num_ops = random.randint(*ops_per_job_range)
        job_ops = []
        for op_id in range(num_ops):
            num_machine_options = random.randint(1, num_machines // 2 + 1)
            machine_options = random.sample(range(num_machines), num_machine_options)
            
            proc_options = []
            for m_id in machine_options:
                proc_time = random.randint(*proc_time_range)
                energy = proc_time * random.uniform(1.0, 2.5)
                carbon = proc_time * random.uniform(0.5, 1.5)
                proc_options.append({
                    "machine": m_id,
                    "processing_time": proc_time,
                    "energy_consumption": round(energy, 2),
                    "carbon_emission": round(carbon, 2)
                })
                total_proc_time += proc_time

            job_ops.append({
                "op_id": op_id,
                "processing_options": proc_options
            })
            
        instance["jobs"].append({
            "job_id": j_id,
            "operations": job_ops
        })

    # --- 设置资源约束 ---
    instance["constraints"] = {
        "max_wip": int(num_jobs * 0.8),
        "total_energy": int(total_proc_time * 1.8),
        "total_carbon": int(total_proc_time * 1.2)
    }
    
    return instance

def main():
    """
    主函数，根据预设配置生成并保存 36 个 FJSP 实例。
    """
    # 定义 36 个算例的配置
    configs = []
    job_sizes = [10, 20, 30, 40]
    machine_sizes = [5, 10, 15]
    
    for j in job_sizes:
        for m in machine_sizes:
            for _ in range(3): # 每个规模生成 3 个实例
                configs.append((j, m))

    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Generating {len(configs)} FJSP Instances ---")
    
    for i, (n_j, n_m) in enumerate(configs):
        instance_data = generate_fjsp_instance(
            num_jobs=n_j,
            num_machines=n_m,
            ops_per_job_range=(3, 8),
            proc_time_range=(10, 50),
            num_transporters=5
        )
        
        # 文件名格式: generated_m{机器数}_j{作业数}_s{序号}.json
        filename = f"generated_m{n_m}_j{n_j}_s{i+1}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(instance_data, f, indent=4)
            
        print(f"  ({i+1}/{len(configs)}) Saved instance to {filepath}")
        
    print("\n--- Instance Generation Finished ---")

if __name__ == "__main__":
    main()
