import heapq
from typing import Dict, List

class Batch:
    def __init__(self, distributor_id, batch_id, ready_time, processing_time, job_ids):
        self.distributor_id = distributor_id
        self.batch_id = batch_id
        self.ready_time = ready_time
        self.processing_time = processing_time
        self.job_ids = job_ids  # 新增：该批次包含的作业ID列表

    def __repr__(self):
        return f"Batch(D{self.distributor_id}-B{self.batch_id}, ready={self.ready_time}, proc={self.processing_time}, jobs={self.job_ids})"

from typing import Dict, List

class DispatchHeuristic:
    def __init__(self):
        pass

    def heuristic_score(self, job, current_time):
        wait_time = max(0, current_time - getattr(job, 'completed_time', 0))
        processing_time = getattr(job, 'processing_time', 1)
        return -(wait_time / (processing_time ** 2))


    def select_action(self, state) -> Dict[str, Dict[int, List[int]]]:
        """
        按 heuristic_score 对作业排序并分批
        返回格式：{'dispatch': {batch_id: [job_id1, job_id2, ...]}}
        """
        completed_jobs = state.get('completed_jobs', [])
        config = state.get('config', None)
        num_windows = getattr(config, 'delivery_num_windows', 2) if config else 2
        time_window = getattr(config, 'delivery_time_window', 8) if config else 8
        current_time = state.get('t', 0)
    
        # 按配送商分组
        distributor_jobs = {}
        for job in completed_jobs:
            distributor_id = getattr(job, 'distributor_id', None)
            if distributor_id is not None:
                distributor_jobs.setdefault(distributor_id, []).append(job)
    
        scheduled_batches = {}
        available_windows = num_windows
        batch_id = 0  # 用于生成全局唯一的批次ID
    
        for distributor_id, jobs in distributor_jobs.items():
            # 按 heuristic_score 降序排序
            jobs.sort(key=lambda j: self.heuristic_score(j, current_time), reverse=True)
            # 再按完成时间分时间窗口分批
            jobs.sort(key=lambda j: getattr(j, 'completed_time', 0))
            grouped = []
            group = []
            group_start = None
            for job in jobs:
                if job.status == 'dispatching' or job.status == 'dispatched':
                    continue
                job_time = getattr(job, 'completed_time', 0)
                if group_start is None:
                    group_start = job_time
                if job_time - group_start <= time_window:
                    group.append(job.job_id)
                else:
                    if group:
                        grouped.append(group)
                        if available_windows <= 0:
                            break
                    group = [job.job_id]
                    group_start = job_time
            if group and available_windows > 0:
                grouped.append(group)
    
            grouped = grouped[:available_windows]
            available_windows -= len(grouped)
    
            # 将分批结果添加到全局批次ID中，批次ID包含配送商信息
            for group in grouped:
                # 批次ID格式: distributor_id * 1000 + 子批次号
                distributor_batch_id = distributor_id * 1000 + len(scheduled_batches)
                scheduled_batches[distributor_batch_id] = group
                batch_id += 1
                
                print(f"创建配送批次: 配送商 {distributor_id}, 批次ID {distributor_batch_id}, 作业数量 {len(group)}")
    
            if available_windows <= 0:
                break
    
        return {'dispatch': scheduled_batches}
