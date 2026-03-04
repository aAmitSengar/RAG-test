import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import certifi

# Fix SSL on macOS by setting environment variable, crucial for `huggingface_hub`
os.environ["SSL_CERT_FILE"] = certifi.where()

# Load token from .env
load_dotenv()
token = os.getenv("HF_TOKEN") or None  # None means unauthenticated (public models only)

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
        if "401" in str(e) or "403" in str(e) or "token" in str(e).lower():
            print(
                "\nAuthentication error. If this model is private or gated, "
                "set a valid HF_TOKEN in your .env file.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    models_dir = base_path / "models"
    models_dir.mkdir(exist_ok=True)

    # Download embedding model
    download_local_model(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(models_dir / "all-MiniLM-L6-v2")
    )

    # Download generation model
    download_local_model(
        repo_id="google-t5/t5-small",
        local_dir=str(models_dir / "t5-small")
    )
