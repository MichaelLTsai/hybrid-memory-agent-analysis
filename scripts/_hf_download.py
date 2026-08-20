"""Fetch one data file from HuggingFace and copy it to a target path.

Called by download_data.sh.

    python _hf_download.py <repo_id> <filename> <destination>
"""
import shutil
import sys

from huggingface_hub import hf_hub_download

repo_id, filename, dest = sys.argv[1], sys.argv[2], sys.argv[3]
src = hf_hub_download(repo_id, filename, repo_type="dataset")
shutil.copy(src, dest)
print(f"    done -> {dest}")
