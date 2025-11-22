# Tech Arena 2025 (Phase 2) - 比赛要求与代码上下文

> **注意**：这是第二阶段 (Phase 2) 的更新文档。包含最新的 GPU 环境、云服务器访问方式及 Mac 本地调试指南。

## 1. 核心目标 (Core Objective)

**任务**: 创建一个高效的 LLM (大语言模型) 推理流程 (Inference Pipeline)，用于单轮问答 (Single-Round QA)。

**科目 (Subjects)**:
- 代数 (Algebra)
- 地理 (Geography)
- 历史 (History)
- **[NEW]** 中文 (Chinese - Language and Culture)

**评分 (Scoring)**:
- **准确度 (Accuracy)**: 60% (由 LLM-as-a-Judge 评估)
- **端到端延迟 (End-to-End Latency)**: 40% (只计算推理时间，不包括加载时间)

**晋级**: 前 8 名队伍将受邀进行演示 (Problem 2)。

## 2. 硬件与环境 (Hardware & Environment)

**最大变化**: GPU 的加入。你的代码必须利用 GPU 进行加速。

- **GPU**: NVIDIA Tesla T4, 16GB VRAM (第一阶段无 GPU)
- **CPU**: Intel x86_64, 16 Cores
- **内存 (Memory)**: 128 GB, DDR4
- **操作系统 (OS)**: Linux
- **语言 (Language)**: Python 3.12
- **核心限制**: **无互联网访问 (No Internet Access)**。评估环境是完全离线的，模型必须从本地缓存加载。

## 3. 允许的模型 (Allowed Models)

你只能使用以下列表中的一个或多个模型（服务器上模型文件位于 `/app/models`）：

- `embeddinggemma-300m`
- `gemma-3-270m-it`
- `gemma-3-1b-it`
- `gemma-3-4b-it`
- **[NEW]** `gemma-3-12b-it`
- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- **[NEW]** `Llama-3.1-8B-Instruct`
- `Qwen3-Embedding-0.6B`
- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `Qwen3-4B`
- **[NEW]** `Qwen3-8B`

## 4. 云服务器访问 (Server Access) [NEW]

我们在 Hackathon 现场，需要远程连接到分配的高性能服务器进行最终测试。

- **IP**: `34.242.203.50`
- **用户**: `user`
- **密钥**: `user.pem` (请确保已下载)

### 连接命令 (Mac Terminal)

**设置权限 (必须)**:
首次使用前，必须修改密钥权限，否则会被拒绝。
```bash
chmod 600 user.pem
```

**SSH 连接**:
```bash
ssh -i user.pem user@34.242.203.50
```

### 文件传输 (SCP)
在本地修改完代码后，将其推送到服务器：

```bash
# 将 inferencePipeline 文件夹上传到服务器的 home 目录
scp -i user.pem -r ./inferencePipeline user@34.242.203.50:~/
```

## 5. 本地调试指南 (MacBook Workflow) [NEW]

在酒店 Wi-Fi 不稳定且环境嘈杂的情况下，利用 MacBook 进行本地逻辑开发，仅用服务器进行最终验证。

### A. 兼容性适配 (CUDA vs MPS)
Mac (M1/M2/M3) 使用 `mps` (Metal) 而不是 `cuda`。为了让同一套代码能在 Mac 和 Linux 服务器上运行，请在 `loadPipeline` 中加入设备判断逻辑：

```python
import torch
import platform

def get_device():
    if torch.cuda.is_available():
        return "cuda"  # 提交到服务器时会走这里
    elif torch.backends.mps.is_available():
        return "mps"   # Mac 本地调试走这里
    else:
        return "cpu"
```

### B. 本地模拟“离线环境”
比赛环境没有网络。在 Mac 上测试时，可以通过代码强制禁止网络访问，以检测是否有隐式下载行为：

```python
import os
# 在 loadPipeline 顶部加入
os.environ['HF_HUB_OFFLINE'] = '1' 
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### C. Mac 开发策略
1.  **轻量级模型测试**: 不要尝试在 Mac 上加载 8B 或 12B 模型（除非你的 Mac 内存很大且你不在乎发热）。下载列表中最小的模型（如 `Qwen3-0.6B` 或 `Llama-3.2-1B`）到本地，用于跑通 pipeline 的输入输出逻辑。
2.  **量化陷阱**: `bitsandbytes` (用于 4-bit/8-bit 量化) 在 Mac 上安装极其麻烦且经常报错。
    *   **建议**: 本地 Mac 调试时使用 `float16` 或 `float32` 跑逻辑。只有在推送到服务器 (Linux + CUDA) 后，再开启 `load_in_4bit=True` 进行显存优化测试。
3.  **Mock 路径**: 本地和服务器路径不同。

```python
# 自动判断路径
base_path = '/app/models' if os.path.exists('/app/models') else './local_models'
```

## 6. 提交结构与执行流程 (Submission)

### 提交内容 (Zip)
`submission.zip` 结构：
```text
submission.zip
└── inferencePipeline/
    ├── __init__.py         # 暴露 loadPipeline
    ├── requirements.txt    # 依赖列表
    └── (其他资源文件)
```
**禁止包含模型文件。**

### 执行流程
1.  **Import**: `from inferencePipeline import loadPipeline`
2.  **Load (不计时)**: `pipeline = loadPipeline()` -> 加载模型到显存。
3.  **Run (计时)**: `answers = pipeline(questions)` -> 核心推理。

## 7. Phase 2 关键优化策略 (Optimization Strategy)

由于引入了 GPU，优化策略与 Phase 1 完全不同：

### 1. 显存管理 (VRAM Management)
Tesla T4 有 16GB 显存。
-   **安全区**: `Qwen3-4B` (约 8-9GB @ fp16)，`Llama-3.2-3B`。
-   **危险区**: `Llama-3.1-8B` 和 `gemma-3-12b`。如果必须使用，必须在服务器上使用 `bitsandbytes` 进行 4-bit 量化 (`load_in_4bit=True`)。

### 2. 批量推理 (Batch Inference)
**极度重要**: 不要用 For 循环！
实现逻辑：
```python
tokenizer(list_of_questions, padding=True, return_tensors='pt').to(device)
model.generate(**inputs)
batch_decode
```

### 3. 针对新科目
-   **中文**: 优先使用 **Qwen**。
