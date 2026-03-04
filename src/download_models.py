import os
import socket
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import certifi


def setup_system_truststore() -> bool:
    """
    Try to route SSL verification through OS trust store.
    Useful on managed macOS environments with enterprise root CAs.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        print("Using system trust store via 'truststore'.")
        return True
    except Exception:
        return False


def setup_ssl_certificates() -> str:
    """
    Configure cert bundle env vars used by requests/urllib/curl-backed clients.
    Returns the cert path used.
    """
    cert_path = certifi.where()
    os.environ["SSL_CERT_FILE"] = cert_path
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    os.environ["CURL_CA_BUNDLE"] = cert_path
    return cert_path


CERT_PATH = setup_ssl_certificates()
USING_SYSTEM_TRUSTSTORE = setup_system_truststore()

# Load token from .env
load_dotenv()
token = os.getenv("HF_TOKEN") or None  # None means unauthenticated (public models only)


def check_connectivity() -> None:
    """Print lightweight diagnostics for DNS/proxy/cert-related failures."""
    print("Connectivity diagnostics:")
    print("  SSL_CERT_FILE:", os.getenv("SSL_CERT_FILE"))
    print("  REQUESTS_CA_BUNDLE:", os.getenv("REQUESTS_CA_BUNDLE"))
    print("  HTTPS_PROXY:", os.getenv("HTTPS_PROXY") or "(not set)")
    print("  HTTP_PROXY:", os.getenv("HTTP_PROXY") or "(not set)")
    print("  NO_PROXY:", os.getenv("NO_PROXY") or "(not set)")
    try:
        ip = socket.gethostbyname("huggingface.co")
        print("  DNS lookup huggingface.co:", ip)
    except Exception as exc:
        print("  DNS lookup huggingface.co failed:", exc)
        print(
            "  Hint: This is a network/DNS issue. Configure DNS or proxy first; "
            "SSL fixes alone will not resolve it."
        )

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
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print(
                "\nSSL certificate validation failed.\n"
                f"Using cert bundle: {CERT_PATH}\n"
                "If this persists on macOS, run your Python 'Install Certificates.command' once.\n"
                "If you are on a managed network, install truststore and retry:\n"
                "  python3 -m pip install --user truststore\n"
            )
        if "401" in str(e) or "403" in str(e) or "token" in str(e).lower():
            print(
                "\nAuthentication error. If this model is private or gated, "
                "set a valid HF_TOKEN in your .env file.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )

if __name__ == "__main__":
    print(f"Using SSL cert bundle: {CERT_PATH}")
    print("System trust store active:", USING_SYSTEM_TRUSTSTORE)
    check_connectivity()
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
