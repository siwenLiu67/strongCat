import gym
from gym import spaces
import numpy as np
import copy
from typing import Dict, Tuple, Optional
import config

EPS = 1e-8

class FJSPEnv(gym.Env):
    """
    用于DRL的柔性作业车间调度问题 (FJSP) 环境。
    动作：选择调度规则（来自 config.SCHEDULING_RULES）。
    时间：事件驱动推进到最近完工时刻。
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, production_data, orders_data, T_internal, normalize_obs: bool = False, seed: Optional[int] = None):
        super().__init__()
        
        self.jobs_data: Dict[str, list] = production_data["jobs"]        # job_id -> [(op_id, {machine_id: time}), ...]
        self.machines_data: Dict[str, dict] = production_data["machines"] # machine_id -> {props...}
        self.precedence: Dict[str, dict] = production_data["precedence"]  # job_id -> {op_id: [pre_op_ids]}
        self.T_internal = float(T_internal)
        self.orders_data = orders_data
        self.normalize_obs = normalize_obs and getattr(config, "NORMALIZE_OBS", False)

        # ---- 可选的显式映射（NEW）----
        self.job_to_market = production_data.get("job_to_market", {})     # job_id -> market
        self.job_to_deadline = production_data.get("job_to_deadline", {}) # job_id -> absolute deadline (float)

        self.num_jobs = len(self.jobs_data)
        self.num_machines = len(self.machines_data)
        self.scheduling_rules = list(config.SCHEDULING_RULES)

        # 稳定的 job 索引映射（CHG）
        self.job_index_map = {jid: i for i, jid in enumerate(self.jobs_data.keys())}

        # 动作空间
        self.action_space = spaces.Discrete(len(self.scheduling_rules))

        # 状态空间定义
        # 机器： [idle(0/1), job_index(-1 or 0..N-1), remaining(proc time)]
        # 工件： [current_op_idx, finished(0/1), remaining_proc_time, due_date]
        # 时间： [current_time, T_internal - current_time]
        state_size = (self.num_machines * 3) + (self.num_jobs * 4) + 2
        self.observation_space = spaces.Box(low=-1.0, high=np.inf, shape=(state_size,), dtype=np.float32)

        # 经验上用于归一化的上界（NEW，可选，不破坏兼容）
        self._max_single_op_time = self._estimate_max_single_op_time()
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        self.reset()

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

    def reset(self):
        self.current_time = 0.0
        self.C_max = 0.0

        # 机器状态
        self.machine_status = {m: {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0}
                               for m in self.machines_data}

        # 完成工序缓存（NEW：加速前置判断）
        self.completed_ops: Dict[Tuple[str, str], float] = {}  # (job_id, op_id) -> end_time

        # 工件状态
        self.job_status = {}
        for job_id, ops in self.jobs_data.items():
            # —— 交付期确定（CHG：优先显式映射，其次 market，最后 T_internal）——
            if job_id in self.job_to_deadline:
                due_date = float(self.job_to_deadline[job_id])
            else:
                market = self.job_to_market.get(job_id)
                if market and market in self.orders_data:
                    due_date = float(self.orders_data[market]['deadline'])
                else:
                    due_date = float(self.T_internal)  # fallback

            self.job_status[job_id] = {
                'current_op_idx': 0,
                'finished': False,
                'release_time': 0.0,
                'due_date': due_date,
                'remaining_proc_time': self._calculate_remaining_proc_time(job_id, 0)
            }

        self.schedule = []
        return self._get_state()

    # --- 工具函数 ---
    def _estimate_max_single_op_time(self) -> float:
        mx = 1.0
        for _, ops in self.jobs_data.items():
            for _, m_times in ops:
                if m_times:
                    mx = max(mx, max(m_times.values()))
        return mx

    def _calculate_remaining_proc_time(self, job_id, op_idx: int) -> float:
        remaining_time = 0.0
        for i in range(op_idx, len(self.jobs_data[job_id])):
            _, machine_times = self.jobs_data[job_id][i]
            if machine_times:
                remaining_time += min(machine_times.values())
        return remaining_time

    # --- 状态构造 ---
    def _get_state(self):
        machine_states = []
        for m in self.machines_data:
            s = self.machine_status[m]
            idle = 1.0 if s['idle'] else 0.0
            job_idx = -1.0
            if s['job'] is not None:
                job_idx = float(self.job_index_map.get(s['job'], -1))
            rem = (s['finish_time'] - self.current_time) if not s['idle'] else 0.0

            if self.normalize_obs:
                # 归一化：job_idx -> [-1, 1] 的近似映射；时间按 T_internal
                job_idx = (job_idx + 1.0) / max(1.0, self.num_jobs)  # [-1,1] 的弱化版（避免复杂映射）
                rem = rem / max(self.T_internal, self._max_single_op_time)

            machine_states.extend([idle, job_idx, rem])

        job_states = []
        for job_id in self.jobs_data:
            st = self.job_status[job_id]
            cur = float(st['current_op_idx'])
            fin = 1.0 if st['finished'] else 0.0
            rpt = float(st['remaining_proc_time'])
            due = float(st['due_date'])

            if self.normalize_obs:
                rpt = rpt / max(self.T_internal, self._max_single_op_time)
                due = due / max(self.T_internal, due + EPS)  # 相对缩放

            job_states.extend([cur, fin, rpt, due])

        time_info = [self.current_time, self.T_internal - self.current_time]
        if self.normalize_obs:
            time_info = [t / max(self.T_internal, EPS) for t in time_info]

        state = np.array(machine_states + job_states + time_info, dtype=np.float32)

        # 健壮性
        if np.isnan(state).any() or np.isinf(state).any():
            print(f"[Env Warning] NaN/Inf in state -> zeroed.")
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state

    # --- 核心交互 ---
    def step(self, action):
        prev_time = self.current_time

        # 在“时间不动且未完成”时循环：优先派工，否则推进时间
        while self.current_time == prev_time and not self._is_done():
            idle_machines = [m for m, s in self.machine_status.items() if s['idle']]

            if not idle_machines:
                self._advance_time()
                continue

            # 构建可选工序池
            all_available_ops = []
            for machine_id in idle_machines:
                ops = self._get_available_ops(machine_id)
                for job_id, op_id, ptime in ops:
                    all_available_ops.append((job_id, op_id, ptime, machine_id))

            if all_available_ops:
                job_id, op_id, proc_time, machine_id = self._apply_scheduling_rule(all_available_ops, action)

                # 派工
                self.machine_status[machine_id] = {
                    'idle': False, 'job': job_id, 'op': op_id, 'finish_time': self.current_time + proc_time
                }

                # 更新工件状态
                st = self.job_status[job_id]
                st['current_op_idx'] += 1
                if st['current_op_idx'] >= len(self.jobs_data[job_id]):
                    st['finished'] = True
                st['remaining_proc_time'] = self._calculate_remaining_proc_time(job_id, st['current_op_idx'])

                self.schedule.append({'job': job_id, 'op': op_id, 'machine': machine_id,
                                      'start': self.current_time, 'end': self.current_time + proc_time})
            else:
                # 无可选工序 -> 推进时间（可能触发死锁检测）
                self._advance_time()

        # 死锁：时间没动且未完成
        if self.current_time == prev_time and not self._is_done():
            self.C_max = float('inf')
            return self._get_state(), -1e6, True, {}

        # 正常奖励与终止判断
        reward = self._calculate_reward()
        done = self._is_done()
        if done:
            self.C_max = self.current_time

        return self._get_state(), reward, done, {}

    def _apply_scheduling_rule(self, available_ops, rule_idx):
        rule_name = self.scheduling_rules[int(rule_idx)]

        if rule_name == 'SPT':   # shortest processing time
            return min(available_ops, key=lambda x: x[2])
        if rule_name == 'LPT':   # longest processing time
            return max(available_ops, key=lambda x: x[2])
        if rule_name == 'FIFO':  # first release time
            return min(available_ops, key=lambda x: self.job_status[x[0]]['release_time'])
        if rule_name == 'EDD':   # earliest due date
            return min(available_ops, key=lambda x: self.job_status[x[0]]['due_date'])
        if rule_name == 'CR':    # critical ratio = (due - now) / remaining_proc_time
            def cr(x):
                due_left = max(self.job_status[x[0]]['due_date'] - self.current_time, EPS)
                remain = max(self.job_status[x[0]]['remaining_proc_time'], EPS)
                return due_left / remain
            return min(available_ops, key=cr)

        raise ValueError(f"未知的调度规则: {rule_name}")

    def _get_available_ops(self, machine_id):
        ops = []
        now = self.current_time
        for job_id, st in self.job_status.items():
            if st['finished'] or now < st['release_time']:
                continue

            op_idx = st['current_op_idx']
            op_id, machine_times = self.jobs_data[job_id][op_idx]

            # 紧前检查（NEW：用 completed_ops O(1)）
            pre_list = self.precedence.get(job_id, {}).get(op_id, [])
            precedents_met = True
            for pre_op_id in pre_list or []:
                if (job_id, pre_op_id) not in self.completed_ops:
                    precedents_met = False
                    break

            if precedents_met and machine_id in machine_times:
                ops.append((job_id, op_id, float(machine_times[machine_id])))

        return ops

    def _advance_time(self):
        # 最近完工时刻
        finish_times = [s['finish_time'] for s in self.machine_status.values() if not s['idle']]
        if not finish_times:
            # 没有忙机，尝试前推到最近 release
            pending_releases = [s['release_time'] for s in self.job_status.values()
                                if not s['finished'] and s['release_time'] > self.current_time]
            if not pending_releases:
                # 无事件可推（可能死锁）
                return

            next_time = min(pending_releases)
        else:
            next_time = min(finish_times)

        # 时间推进
        self.current_time = next_time

        # 机器释放 + 完成缓存（NEW）
        for m_id, ms in self.machine_status.items():
            if not ms['idle'] and ms['finish_time'] <= self.current_time + EPS:
                finished_job_id = ms['job']
                finished_op_id = ms['op']
                self.completed_ops[(finished_job_id, finished_op_id)] = ms['finish_time']

                self.machine_status[m_id] = {'idle': True, 'job': None, 'op': None, 'finish_time': -1.0}

                # 下道工序解锁
                if not self.job_status[finished_job_id]['finished']:
                    self.job_status[finished_job_id]['release_time'] = self.current_time

    def _calculate_reward(self):
        # 中间奖励：每次 step 至少完成一道工序 -> 给 +1
        reward = 1.0

        if self._is_done():
            C_max = self.current_time
            if C_max > self.T_internal:
                reward = - (config.K + config.ALPHA * (C_max - self.T_internal))
            else:
                reward = config.BETA * (self.T_internal - C_max)
        return float(reward)

    def _is_done(self):
        return all(st['finished'] for st in self.job_status.values())

    def render(self, mode='human'):
        print(f"time={self.current_time:.4f}, C_max={self.C_max}")
        print("machines:", self.machine_status)
        print("jobs:", self.job_status)
        print("schedule:", self.schedule)
