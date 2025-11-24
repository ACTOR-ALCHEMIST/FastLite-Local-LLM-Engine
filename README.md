# Tech Arena 2025 Phase 2 - Inference Pipeline 🏆

> **2nd Place Winner** in the Tech Arena 2025 Phase 2 Competition.

This project implements a high-performance inference pipeline designed for the Tech Arena 2025 competition. It leverages the **Qwen3-4B** model with advanced optimization techniques to achieve a balance between speed and accuracy, tailored for T4 GPU environments.

## ✨ Key Features

*   **Adaptive Complexity Detection**: Automatically categorizes queries into `simple`, `medium`, or `complex` to dynamically adjust inference parameters (batch size, max tokens).
*   **Dynamic Batching**: Optimizes throughput by grouping queries of similar complexity.
*   **Model Optimization**: Supports 4-bit/8-bit quantization (via `BitsAndBytesConfig`) for efficient memory usage on T4 GPUs.
*   **Prompt Engineering**: Implements direct-answer prompts and judge-guided injection to maximize scoring potential.
*   **Automated Deployment**: Includes a one-click script for seamless synchronization and testing on cloud servers.

## 🚀 Quick Start

### 1. Prerequisites
Ensure your `user.pem` key file is located in the project root.

### 2. One-Click Deploy & Test
Run the following command to sync code to the server and execute the test suite:

```bash
./deploy_and_test.sh
```

**What this script does:**
1.  Sets correct permissions for the key.
2.  Syncs `inferencePipeline/` and `run.py` to the remote server.
3.  Executes the inference pipeline and streams logs back to your local terminal.

## � Project Structure

*   `inferencePipeline/`: Core inference logic.
    *   `load.py`: **[Core]** Model loading, complexity detection, and batch inference implementation.
*   `run.py`: Entry point for testing with a predefined question bank.
*   `deploy_and_test.sh`: Automated deployment script.

## 🛠 Maintenance

*   **Model Logic**: Modify `inferencePipeline/load.py` to adjust the model, prompt strategies, or generation configs.
*   **Test Cases**: Update `run.py` to add or remove questions in `QUESTION_BANK` for rapid validation.

## 📝 Notes

*   **Environment**: Optimized for Cloud Linux environments with T4 GPUs.
*   **Monitoring**: Real-time logs are provided via the deployment script for immediate feedback.

---
*Developed for Tech Arena 2025.*
