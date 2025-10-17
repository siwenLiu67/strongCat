import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from typing import List, Dict, Tuple
import requests

class DeepSeekScheduler:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b"):
        """
        使用DeepSeek模型作为调度求解器
        可以选择: deepseek-ai/deepseek-coder-1.3b, deepseek-ai/deepseek-llm-7b-chat 等
        """
        print("正在加载DeepSeek模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("DeepSeek模型加载完成!")
    
    def create_thinking_prompt(self, problem_text: str) -> str:
        """创建带有思维链的提示词"""
        prompt = f"""你是一个组合优化专家，请解决以下车间调度问题。请按步骤思考：

问题描述:
{problem_text}

请按以下步骤推理：
1. 首先分析机器资源和工件工序
2. 识别关键约束和潜在冲突
3. 确定最优调度策略
4. 生成详细的调度方案

约束条件:
- 每台机器同一时间只能加工一个工件
- 工件的工序必须按顺序执行
- 目标是最小化总完成时间

请输出以下格式：

推理过程:
[你的逐步推理]

调度方案:
时间 0-X: 机器A加工工件P, 机器B加工工件Q
时间 X-Y: 机器A加工工件R, 机器B加工工件S
...
总完成时间：Z小时

现在开始分析：
"""
        return prompt
    
    def simulate_sft_training(self, training_examples: List[Tuple[str, str]]):
        """模拟监督微调过程"""
        print("正在进行SFT训练...")
        
        for i, (problem, optimal_solution) in enumerate(training_examples):
            prompt = self.create_thinking_prompt(problem)
            full_training_text = prompt + optimal_solution
            
            # 模拟训练过程 - 实际中这里会是真正的训练循环
            inputs = self.tokenizer(
                full_training_text, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if i % 10 == 0:
                print(f"SFT训练进度: {i+1}/{len(training_examples)}")
        
        print("SFT训练完成!")
    
    def generate_with_deepseek(self, prompt: str, max_length=512) -> str:
        """使用DeepSeek模型生成解决方案"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 将输入移到GPU
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def extract_solution(self, generated_text: str) -> str:
        """从生成的文本中提取调度方案"""
        if "调度方案:" in generated_text:
            return generated_text.split("调度方案:")[1].strip()
        elif "调度方案" in generated_text:
            return generated_text.split("调度方案")[1].strip()
        else:
            # 返回最后一段作为方案
            lines = generated_text.split('\n')
            solution_lines = []
            for line in lines:
                if any(keyword in line for keyword in ['时间', '加工', '总完成时间']):
                    solution_lines.append(line)
            return '\n'.join(solution_lines) if solution_lines else generated_text
    
    def evaluate_solution(self, problem: str, solution: str) -> Dict:
        """评估解决方案质量"""
        # 提取关键信息
        machines = re.findall(r'M\d', problem)
        jobs = re.findall(r'J\d', problem)
        
        # 检查方案完整性
        completeness_score = self._check_completeness(solution, jobs)
        
        # 检查格式正确性
        format_score = self._check_format(solution)
        
        # 检查时间连续性
        time_score = self._check_time_continuity(solution)
        
        total_score = (completeness_score + format_score + time_score) / 3
        
        return {
            "completeness": completeness_score,
            "format": format_score, 
            "time_continuity": time_score,
            "total_score": total_score
        }
    
    def _check_completeness(self, solution: str, jobs: List[str]) -> float:
        """检查是否所有工件都被调度"""
        found_jobs = []
        for job in jobs:
            if job in solution:
                found_jobs.append(job)
        return len(found_jobs) / len(jobs) if jobs else 0.5
    
    def _check_format(self, solution: str) -> float:
        """检查输出格式规范性"""
        score = 0
        if "时间" in solution:
            score += 0.3
        if "加工" in solution:
            score += 0.3
        if "总完成时间" in solution:
            score += 0.4
        return score
    
    def _check_time_continuity(self, solution: str) -> float:
        """检查时间连续性"""
        time_pattern = r'时间 (\d+)-(\d+)'
        matches = re.findall(time_pattern, solution)
        
        if len(matches) < 2:
            return 0.5
        
        # 检查时间是否连续
        prev_end = 0
        continuity_violations = 0
        
        for start_str, end_str in matches:
            start, end = int(start_str), int(end_str)
            if start != prev_end:
                continuity_violations += 1
            prev_end = end
        
        continuity_score = 1 - (continuity_violations / len(matches))
        return max(continuity_score, 0)

class DeepSeekTrainingSimulator:
    def __init__(self):
        self.scheduler = DeepSeekScheduler()
    
    def create_training_data(self) -> List[Tuple[str, str]]:
        """创建训练数据"""
        training_examples = [
            (
                "车间有2台机器（M1, M2）需要加工2个工件：J1在M1上加工2小时；J2在M2上加工3小时",
                """推理过程:
1. 分析资源：2台机器(M1,M2)，2个工件(J1,J2)
2. 约束：无工序顺序要求，无资源冲突
3. 策略：并行加工两个工件
4. 优化：总完成时间由最长工序决定

调度方案:
时间 0-2: M1加工J1, M2加工J2
总完成时间：3小时"""
            ),
            (
                "车间有2台机器（M1, M2）需要加工3个工件：J1在M1上加工2小时；J2在M2上加工1小时；J3在M1上加工1小时",
                """推理过程:
1. 分析：M1需要处理J1(2h)和J3(1h)，M2处理J2(1h)
2. 约束：M1不能同时加工两个工件
3. 策略：先加工J1，然后J3；M2单独加工J2
4. 优化：总时间3小时

调度方案:
时间 0-2: M1加工J1, M2加工J2
时间 2-3: M1加工J3, M2空闲
总完成时间：3小时"""
            )
        ]
        return training_examples
    
    def train_and_evaluate(self):
        """完整的训练和评估流程"""
        print("=" * 60)
        print("DeepSeek调度求解器训练演示")
        print("=" * 60)
        
        # 1. 准备训练数据
        training_data = self.create_training_data()
        print(f"准备 {len(training_data)} 个训练样本")
        
        # 2. SFT训练
        self.scheduler.simulate_sft_training(training_data)
        
        # 3. 测试新问题
        test_problems = [
            "车间有2台机器（M1, M2）需要加工3个工件：J1在M1上加工2小时然后在M2上加工1小时；J2在M2上加工3小时；J3在M1上加工1小时",
            "调度问题：2台机器(M1,M2)，2个工件：J1在M1上加工3小时；J2在M1上加工1小时然后在M2上加工2小时",
            "制造车间：机器M1和M2，工件J1(M1:2h)、J2(M2:2h,M1:1h)、J3(M2:1h)"
        ]
        
        print("\n" + "=" * 60)
        print("测试DeepSeek模型求解新问题")
        print("=" * 60)
        
        for i, problem in enumerate(test_problems, 1):
            print(f"\n问题 {i}: {problem}")
            print("-" * 50)
            
            # 生成提示词
            prompt = self.scheduler.create_thinking_prompt(problem)
            
            # 使用DeepSeek生成解决方案
            print("DeepSeek正在推理...")
            generated_text = self.scheduler.generate_with_deepseek(prompt)
            
            # 提取方案
            solution = self.scheduler.extract_solution(generated_text)
            
            print("生成的解决方案:")
            print(solution)
            
            # 评估方案
            evaluation = self.scheduler.evaluate_solution(problem, solution)
            print(f"\n评估结果: 总分{evaluation['total_score']:.2f} "
                  f"(完整性{evaluation['completeness']:.2f}, "
                  f"格式{evaluation['format']:.2f}, "
                  f"时间连续性{evaluation['time_continuity']:.2f})")
            
            print("-" * 50)

def main():
    # 运行完整的训练和测试流程
    simulator = DeepSeekTrainingSimulator()
    simulator.train_and_evaluate()
    
    # 演示直接使用DeepSeek求解
    print("\n" + "=" * 60)
    print("DeepSeek直接求解演示")
    print("=" * 60)
    
    direct_scheduler = DeepSeekScheduler()
    
    new_problem = """车间有3台机器（M1, M2, M3）需要加工4个工件：
J1: 在M1上加工2小时
J2: 在M2上加工3小时然后在M1上加工1小时  
J3: 在M3上加工2小时
J4: 在M2上加工1小时然后在M3上加工1小时"""
    
    prompt = direct_scheduler.create_thinking_prompt(new_problem)
    print("问题:", new_problem)
    print("\nDeepSeek正在思考...")
    
    solution = direct_scheduler.generate_with_deepseek(prompt)
    print("\nDeepSeek完整输出:")
    print(solution)

if __name__ == "__main__":
    main()