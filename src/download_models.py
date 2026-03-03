import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import certifi

# Fix SSL on macOS by setting environment variable, crucial for `huggingface_hub`
os.environ["SSL_CERT_FILE"] = certifi.where()

# Load token from .env
load_dotenv()
token = os.getenv("HF_TOKEN")

def download_local_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    
    # Download embedding model
    download_local_model(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(base_path / "all-MiniLM-L6-v2")
    )
    
    # Download generation model
    download_local_model(
        repo_id="t5-small",
        local_dir=str(base_path / "t5-small")
    )
