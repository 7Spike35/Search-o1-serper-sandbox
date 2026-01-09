from huggingface_hub import snapshot_download
import os

# 1. 设置镜像（如果在国内）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 纯文件下载，不进行读取检查
print("开始下载文件...")
snapshot_download(
    repo_id="miromind-ai/MiroThinker-v1.0-8B",
    repo_type="model",
    local_dir="./models/MiroThinker-8B",    # 下载到当前目录下的文件夹
    local_dir_use_symlinks=False,    # 下载真实文件
    resume_download=True             # 支持断点续传
)
print("下载完成")