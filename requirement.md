# Tech Arena 2025 - 比赛要求与代码上下文
[比赛完整要求（可做上下文）](requirement.md)

## 1. 核心目标 (Core Objective)

- **任务:** 创建一个高效的 LLM (大语言模型) 推理流程 (Inference Pipeline)，用于**单轮问答 (Single-Round QA)**。
- **主题:** 答案涵盖三个领域：**代数 (Algebra)**、**地理 (Geography)** 和 **历史 (History)**。
- **评分:**
  - **准确度 (Accuracy):** 60% (由 LLM-as-a-Judge 评估)
  - **端到端延迟 (End-to-End Latency):** 40% (只计算推理时间，不包括加载时间)
- **门槛:** 必须达到 10% 的最低准确度才能进入排行榜。

## 2. 硬件与环境 (Hardware & Environment)

- **CPU:** AMD x86_64, 16 Cores
- **内存 (Memory):** 128 GB, DDR4
- **操作系统 (OS):** Linux
- **语言 (Language):** Python 3.12
- **核心限制:** **无互联网访问 (No Internet Access)**。评估环境是完全离线的。

## 3. 允许的模型 (Allowed Models)

你只能使用以下列表中的一个或多个模型：

1. `embeddinggemma-300m`
2. `gemma-3-270m-it`
3. `gemma-3-1b-it`
4. `gemma-3-4b-it`
5. `Llama-3.2-1B-Instruct`
6. `Llama-3.2-3B-Instruct`
7. `Qwen3-Embedding-0.6B`
8. `Qwen3-0.6B`
9. `Qwen3-1.7B`
10. `Qwen3-4B`

## 4. 提交结构与执行流程 (Submission Structure & Execution Flow)

### 提交内容

你需要提交一个 `.zip` 文件，其内部结构必须如下：

```
submission.zip
└── inferencePipeline/
    ├── __init__.py         # (或其他你的 .py 代码文件, 比如 load.py)
    ├── requirements.txt    # (所有依赖)
    └── (其他必要文件, 如 RAG 索引库, 数据库等)
```

- **禁止:** 不得包含 LLM 模型文件。
- **大小:** 压缩包最大 1 GB。

### 执行流程

评估系统将使用一个类似 `run.py` 的脚本来运行你的代码：

1. **导入:** `from inferencePipeline import loadPipeline`
   - 这会导入你 `inferencePipeline` 文件夹中的 `loadPipeline` 函数。
2. **加载 (不计时):** `pipeline = loadPipeline()`
   - 调用此函数来加载你的模型、RAG 索引、数据库等所有资源。
   - 此步骤的**时间不计入延迟**。
3. **推理 (计时):** `answers = pipeline(questions)`
   - 调用上一步返回的 `pipeline` 对象，并传入问题列表。
   - **此步骤的执行时间是你的延迟分数**。

## 5. 关键代码要求 (Critical Code Requirements)

### `inferencePipeline` 文件夹

- 你所有的代码、逻辑和资源文件（模型除外）都必须位于此文件夹内。

### `loadPipeline()` 函数 (入口)

- 你必须在 `inferencePipeline` 模块中（例如，放在 `inferencePipeline/__init__.py` 或 `inferencePipeline/load.py` 中并在 `__init__.py` 导入）提供一个 `loadPipeline()` 函数。
- 此函数不接受任何参数。
- 它必须加载模型、初始化所有资源（如 RAG 索引）。
- 它必须返回一个**可调用对象 (Callable Object)**。最简单的实现是返回一个类的实例（如示例中的 `MyModel` 实例）。

### 返回的可调用对象 (推理逻辑)

- 这个对象 (例如 `MyModel` 实例) 必须能像函数一样被调用，并接受一个参数：`questions` 列表。

- **输入 `questions` 格式:**

  ```
  [
      {"questionID": 301, "question": "What is the capital of Greece?"},
      {"questionID": 302, "question": "What is the capital of Thailand?"},
      ...
  ]
  ```

- **输出 `answers` 格式:**

  ```
  [
      {"questionID": 301, "answer": "The capital of Greece is Athens."},
      {"questionID": 302, "answer": "The capital of Thailand is Bangkok."},
      ...
  ]
  ```

- **约束:**

  - `questionID` 必须准确无误地传回。
  - `answer` 必须是一个字符串。
  - 答案字符串长度**不得超过 5000 个字符**。

### 离线与模型加载

- **必须**设置环境变量以强制离线：

  ```
  import os
  os.environ['HF_HUB_OFFLINE'] = '1'
  os.environ['TRANSFORMERS_OFFLINE'] = '1'
  ```

- **必须**从指定的本地缓存路径加载模型：

  ```
  cache_dir = '/app/models'
  
  tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
  model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, ...)
  ```

## 6. 如何扩展与保持无 Bug (How to Expand & Stay Bug-Free)

`load.py` 文件是一个很好的起点。以下是如何在其基础上扩展并确保稳定：

### 扩展思路 (基于 `load.py`)

- **`loadPipeline()` (加载阶段 - 不计时):**
  - 这是你进行所有重度操作的地方。
  - 加载模型 (`get_model()`)。
  - 如果你使用 RAG (Retrieval-Augmented Generation) 来提高准确性，请在此处加载或构建你的向量索引（例如 Faiss, ChromaDB）。
  - 如果你使用多个模型（例如，一个用于代数，一个用于历史/地理），请在此处全部加载它们。
- **`__call__(self, questions)` (推理阶段 - 计时):**
  - **关键优化：批量处理 (Batching)**
    - `load.py` 示例使用 for 循环**逐个**处理问题，效率极低。
    - 你应该修改 `__call__`：
      1. 将所有 `questions` 列表中的问题文本提取出来，构建成一个**批量 (batch)**。
      2. 使用 `tokenizer` 一次性处理这个批量：`inputs = tokenizer(batch_prompts, padding=True, return_tensors='pt')`。
      3. 使用 `model.generate(inputs, ...)` 一次性生成所有答案。
      4. 解码批量结果，然后重新组装成 `answers` 列表格式。
  - **RAG 集成:**
    1. (批量) 对问题进行 embedding。
    2. (批量) 搜索 RAG 索引。
    3. 将检索到的上下文 (context) 与原始问题 (question) 组装成更丰富的 prompt。
    4. 将这些 prompts 批量送入 LLM。
  - **模型路由 (Model Routing):**
    - 在此处添加逻辑，判断问题类型（例如基于关键词 "Solve for x" 判断为代数），并将该问题路由到最合适的（可能是更小的）模型或 prompt 模板。

### 保持无 Bug (Bug-Free Strategy)

1. **严格遵守格式:** 确保 `questionID` 始终正确传递，并且 `answer` 始终是字符串。
2. **异常处理:** `load.py` 中的 `try...except` 块很好。保留它，确保**单个问题失败不会导致整个批量或流程崩溃**。如果单个问题失败，为其返回一个空答案 `''`。
3. **本地测试:** 严格使用官方提供的 `run.py` 文件在**文件夹外**测试你的 `inferencePipeline` 文件夹。确保 `from inferencePipeline import loadPipeline` 能成功导入。
4. **依赖管理:** `requirements.txt` 必须包含所有你 `import` 的库。
5. **离线测试:** 在本地**断开网络连接**后，运行 `run.py`，确保你的代码不会尝试任何网络调用。