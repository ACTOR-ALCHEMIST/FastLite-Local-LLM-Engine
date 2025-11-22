# Tech Arena 2025 (Phase 2) - Competition Requirements & Context

> **Note**: This is the updated documentation for Phase 2. It includes the latest GPU environment details, cloud server access methods, and Mac local debugging guides.

## 1. Core Objective

**Task**: Create an efficient LLM (Large Language Model) Inference Pipeline for Single-Round QA.

**Subjects**:
- Algebra
- Geography
- History
- **[NEW]** Chinese (Language and Culture)

**Scoring**:
- **Accuracy**: 60% (Evaluated by LLM-as-a-Judge)
- **End-to-End Latency**: 40% (Inference time only, excludes loading time)

**Qualification**: Top 8 teams will be invited for presentation (Problem 2).

## 2. Hardware & Environment

**Major Change**: Introduction of GPU. Your code **must** utilize the GPU for acceleration.

- **GPU**: NVIDIA Tesla T4, 16GB VRAM (Phase 1 had no GPU)
- **CPU**: Intel x86_64, 16 Cores
- **Memory**: 128 GB, DDR4
- **OS**: Linux
- **Language**: Python 3.12
- **Core Constraint**: **No Internet Access**. The evaluation environment is completely offline; models must be loaded from local cache.

## 3. Allowed Models

You may only use one or more models from the following list (Model files on server located at `/app/models`):

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

## 4. Cloud Server Access [NEW]

We are at the Hackathon site and need to remotely connect to the assigned high-performance server for final testing.

- **IP**: `34.242.203.50`
- **User**: `user`
- **Key**: `user.pem` (Ensure this is downloaded)

### Connection Command (Mac Terminal)

**Set Permissions (Mandatory)**:
Before first use, modify key permissions or connection will be refused.
```bash
chmod 600 user.pem
```

**SSH Connection**:
```bash
ssh -i user.pem user@34.242.203.50
```

### File Transfer (SCP)
After modifying code locally, push it to the server:

```bash
# Upload inferencePipeline folder to server home directory
scp -i user.pem -r ./inferencePipeline user@34.242.203.50:~/
```

## 5. Local Debugging Guide (MacBook Workflow) [NEW]

Given unstable hotel Wi-Fi and noisy environment, use MacBook for local logic development and only use the server for final verification.

### A. Compatibility Adaptation (CUDA vs MPS)
Mac (M1/M2/M3) uses `mps` (Metal) instead of `cuda`. To ensure the same code runs on both Mac and Linux Server, add device detection logic in `loadPipeline`:

```python
import torch
import platform

def get_device():
    if torch.cuda.is_available():
        return "cuda"  # Server submission path
    elif torch.backends.mps.is_available():
        return "mps"   # Local Mac debugging path
    else:
        return "cpu"
```

### B. Local Simulation of "Offline Environment"
The competition environment has no network. Test on Mac by forcibly disabling network access in code to detect implicit downloads:

```python
import os
# Add to top of loadPipeline
os.environ['HF_HUB_OFFLINE'] = '1' 
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

### C. Mac Development Strategy
1.  **Lightweight Model Testing**: Do not attempt to load 8B or 12B models on Mac (unless you have high RAM and don't mind heat). Download the smallest models (e.g., `Qwen3-0.6B` or `Llama-3.2-1B`) locally to verify pipeline input/output logic.
2.  **Quantization Pitfalls**: `bitsandbytes` (for 4-bit/8-bit quantization) is problematic on Mac.
    *   **Recommendation**: Use `float16` or `float32` for logic debugging on Mac. Only enable `load_in_4bit=True` for VRAM optimization after pushing to the server (Linux + CUDA).
3.  **Mock Paths**: Local and server paths differ.

```python
# Auto-detect path
base_path = '/app/models' if os.path.exists('/app/models') else './local_models'
```

## 6. Submission Structure & Execution Flow

### Submission Content (Zip)
`submission.zip` structure:
```text
submission.zip
└── inferencePipeline/
    ├── __init__.py         # Exposes loadPipeline
    ├── requirements.txt    # Dependency list
    └── (Other resource files)
```
**Do NOT include model files.**

### Execution Flow
1.  **Import**: `from inferencePipeline import loadPipeline`
2.  **Load (Not Timed)**: `pipeline = loadPipeline()` -> Loads model into VRAM.
3.  **Run (Timed)**: `answers = pipeline(questions)` -> Core inference.

## 7. Phase 2 Key Optimization Strategy

Since GPU is introduced, optimization differs completely from Phase 1:

### 1. VRAM Management
Tesla T4 has 16GB VRAM.
-   **Safe Zone**: `Qwen3-4B` (approx 8-9GB @ fp16), `Llama-3.2-3B`.
-   **Danger Zone**: `Llama-3.1-8B` and `gemma-3-12b`. If used, you **must** use `bitsandbytes` for 4-bit quantization (`load_in_4bit=True`) on the server.

### 2. Batch Inference
**Extremely Important**: Do NOT use For loops!
Implementation logic:
```python
tokenizer(list_of_questions, padding=True, return_tensors='pt').to(device)
model.generate(**inputs)
batch_decode
```

### 3. New Subject Strategy
-   **Chinese**: Prioritize using **Qwen**.
