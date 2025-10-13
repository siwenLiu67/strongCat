# FJSP 算法比较框架 (Algorithm Comparison Framework for FJSP)

这是一个用于解决灵活作业车间调度问题 (Flexible Job-Shop Scheduling Problem, FJSP) 并比较不同求解算法性能的 Python 框架。

该项目旨在提供一个模块化的、可扩展的、易于配置的平台，用于测试和评估包括元启发式算法和深度强化学习在内的各种优化算法。

## ✨ 特性 (Features)

- **配置驱动**: 所有实验参数、算法超参数和路径都在唯一的 `config.py` 文件中进行管理，无需修改代码即可调整实验。
- **模块化设计**: 清晰分离的模块，包括环境 (`models`)、算法 (`comparison_algorithms`)、实验 (`experiments`) 和实例 (`instances`)。
- **可扩展的算法库**: 基于抽象基类 (`BaseAlgorithm`) 构建，可以轻松添加新的自定义算法。
- **并行处理**: 利用多核 CPU 并行运行不同实例的实验，大大缩短了总实验时间。
- **自动化分析**: 实验完成后，会自动运行分析脚本，生成性能摘要和多种可视化图表。
- **进度可视化**: 使用 `tqdm` 在命令行中为实验过程提供实时的进度条。

## 📂 项目结构 (Project Structure)

```
.
├── config.py                   # 唯一的中央配置文件
├── comparison_algorithms/      # 存放所有求解算法
│   ├── base_algorithm.py       # 所有算法的抽象基类
│   ├── metaheuristic_algorithms/ # 元启发式算法 (GA, SA, PSO)
│   └── single_layer_drl/       # 深度强化学习算法 (DQN, A2C, PPO)
├── experiments/                # 实验和分析脚本
│   ├── run_fjsp_comparison.py  # 主实验运行脚本
│   └── analyzer/
│       └── analyze_results.py  # 结果分析和可视化脚本
├── instances/                  # 存放 FJSP 实例文件
│   └── instance_generator.py   # 用于生成新实例的脚本
├── models/                     # 仿真环境模型
│   └── fjsp_env.py             # FJSP 环境的核心实现
├── results/                    # 存放实验结果
│   ├── all_results.csv         # 原始的 CSV 结果数据
│   └── figures/                # 生成的分析图表
└── README.md                   # 本文档
```

## 🚀 如何使用 (How to Use)

### 1. 安装依赖 (Install Dependencies)

首先，请确保您已经安装了所有必需的 Python 库。

```bash
pip install -r requirements.txt
```
*(注意: `requirements.txt` 可能需要您手动创建，主要包含 `numpy`, `pandas`, `torch`, `tqdm`, `matplotlib`, `seaborn`)*

### 2. 配置实验 (Configure the Experiment)

打开 `config.py` 文件。在这里，您可以轻松地调整所有设置：

- **`NUM_PROCESSES`**: 修改用于并行处理的 CPU 核心数。
- **`INSTANCE_GLOB_PATTERN`**: 定义您想要运行的实例文件（例如，`"generated_m10_*.json"` 只运行10台机器的实例）。
- **算法超参数**: 在 `DQN_CONFIG`, `GA_CONFIG`, `SA_CONFIG`, `PSO_CONFIG` 等字典中微调每个算法的超参数。

### 3. 运行实验和分析 (Run Experiment and Analysis)

只需在项目根目录下执行以下单个命令，即可启动整个自动化流程：

```bash
python experiments/run_fjsp_comparison.py
```

脚本将自动：
1.  查找所有匹配的实例文件。
2.  并行运行所有已实现的算法。
3.  将所有结果保存到 `results/all_results.csv`。

### 4. (可选) 单独运行分析 (Run Analysis Separately)

如果您想在实验结束后重新运行分析，或者对已有的 `all_results.csv` 文件进行分析，可以单独运行分析脚本：

```bash
python experiments/analyzer/analyze_results.py
```

### 5. 查看结果 (View Results)

- **原始数据**: 所有的原始性能数据都保存在 `results/all_results.csv` 文件中。
- **性能摘要**: 分析脚本会在命令行中打印出各算法的平均性能摘要。
- **可视化图表**: 所有生成的图表（`.png` 文件）都保存在 `results/figures/` 目录中，包括：
    - `overall_performance_boxplot.png`: 总体性能箱线图。
    - `performance_by_instance_type.png`: 按实例类型分组的性能对比。
    - `execution_time_comparison.png`: 算法执行时间对比。

## 🛠️ 如何扩展 (How to Extend)

### 添加新算法

1.  在 `comparison_algorithms` 目录下创建一个新的 Python 文件。
2.  创建一个继承自 `BaseAlgorithm` 的新类。
3.  实现 `__init__` 和 `solve` 方法。
4.  在 `config.py` 中为您的新算法添加一个配置字典。
5.  在 `experiments/run_fjsp_comparison.py` 的 `get_all_agents` 函数中，导入并实例化您的新算法。

---
*这个框架为您提供了一个坚实的基础，以进行高效、可重复的计算实验。*
