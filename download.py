from huggingface_hub import hf_hub_download
import os

repo_id = "AutonLab/MOMENT-1-small"
local_dir = "./MOMENT-1-small"

# 需要下载的核心文件列表
files_to_download = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]

for file in files_to_download:
    print(f"Downloading {file}...")
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)

print("Download complete!")