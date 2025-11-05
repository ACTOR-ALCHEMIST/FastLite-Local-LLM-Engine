# Hackathon Huawei Inference Pipeline / 华为黑客松推理流水线

## Quick Start / 快速开始
- Clone the repo and enter the folder. / 克隆仓库后进入项目目录。
  ```bash
  git clone <repo-url>
  cd inferencePipeline
  ```
- Create and activate the Python environment. / 创建并激活 Python 虚拟环境。
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Run the pipeline entry point to confirm model loading. / 运行入口脚本验证模型加载。
  ```bash
  python load.py
  ```
## Running Successful output / 成功运行截图
![terminal output](./image/Screenshot%202025-10-30%20at%2019.48.23.png)
## Development Notes / 开发注意事项
- Models download to `./app/models`; keep the folder out of Git commits by using `.gitignore`. / 模型会下载到 `./app/models`，通过 `.gitignore` 避免提交到 Git。
- Keep environment variables such as `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` aligned with local needs. / 根据本地需求调整 `HF_HUB_OFFLINE` 和 `TRANSFORMERS_OFFLINE` 等环境变量。
- Follow minimal change guidelines and annotate local tweaks with `# Jiang:` comments as needed. / 尽量保持修改最小化，并在需要时使用 `# name:` 注释标记谁进行的改动。
