#!/usr/bin/env python3
"""
Merge system CA + certifi, test HTTPS to huggingface, then run snapshot_download.
Run: python3 download_with_merged_ca.py
"""
import os
import sys
import time
import traceback
from pathlib import Path

# === Configuration ===
TARGET_ROOT = Path.home() / "practice" / "RAG" / "RAG-test" / "models"
T5_DIR = TARGET_ROOT / "t5-small"
MINILM_DIR = TARGET_ROOT / "all-MiniLM-L6-v2"

# Candidate system CA paths (macOS / common locations)
SYSTEM_CA_CANDIDATES = [
    "/etc/ssl/cert.pem",                    # what curl used earlier
    "/usr/local/etc/openssl@3/cert.pem",   # mac/homebrew openssl
    "/etc/ssl/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
    "/usr/local/ssl/certs/ca-bundle.crt",
]

MERGED_BUNDLE = Path.home() / ".cache" / "merged-ca-bundle.pem"
RETRIES = 3
RETRY_DELAY = 5

def find_system_ca():
    for p in SYSTEM_CA_CANDIDATES:
        if Path(p).exists():
            return p
    return None

def create_merged_bundle(system_ca_path, merged_path):
    import certifi
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "wb") as out:
        if system_ca_path:
            with open(system_ca_path, "rb") as f:
                out.write(f.read())
                out.write(b"\n")
        with open(certifi.where(), "rb") as f:
            out.write(f.read())
    return str(merged_path)

def test_requests_with_bundle(bundle_path):
    import requests
    url = "https://huggingface.co/t5-small/resolve/main/tokenizer_config.json"
    print("Testing requests.get to", url, "using verify=", bundle_path)
    r = requests.get(url, timeout=15, verify=bundle_path)
    print("status:", r.status_code)
    print("first 200 bytes:", r.content[:200])
    return r.status_code == 200

def snapshot_download_with_env(repo_id, cache_dir):
    # Ensure huggingface_hub is available
    try:
        import huggingface_hub
    except Exception:
        print("Installing huggingface_hub and related packages...")
        os.execvp(sys.executable, [sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub", "transformers", "safetensors", "sentence-transformers"])

    # after installation the process will be replaced; unreachable after exec in that case
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo_id, cache_dir=str(cache_dir), force_download=True, resume_download=True)

def main():
    print("Target directories:")
    print(" T5_DIR:", T5_DIR)
    print(" MINILM_DIR:", MINILM_DIR)
    T5_DIR.mkdir(parents=True, exist_ok=True)
    MINILM_DIR.mkdir(parents=True, exist_ok=True)

    system_ca = find_system_ca()
    if system_ca:
        print("Found system CA at:", system_ca)
    else:
        print("WARNING: No system CA found in candidates. You may need to export your corporate CA or provide path.")
        print("Candidates checked:", SYSTEM_CA_CANDIDATES)

    merged = create_merged_bundle(system_ca, MERGED_BUNDLE)
    print("Merged CA bundle written to:", merged)

    # Set env for this process (and child processes)
    os.environ["SSL_CERT_FILE"] = merged
    os.environ["REQUESTS_CA_BUNDLE"] = merged
    print("Set SSL_CERT_FILE and REQUESTS_CA_BUNDLE to merged bundle for this process.")

    # Quick test with requests using explicit verify=merged
    try:
        ok = test_requests_with_bundle(merged)
        if not ok:
            print("requests test did not return 200. Aborting download.")
            return 1
    except Exception as e:
        print("requests test failed with exception:")
        traceback.print_exc()
        print("If this fails, ensure the merged bundle includes your corporate CA (Zscaler) or your system CA path is correct.")
        return 1

    # Try snapshot downloads with retries
    for repo_id, target in [("t5-small", T5_DIR), ("sentence-transformers/all-MiniLM-L6-v2", MINILM_DIR)]:
        attempt = 0
        while attempt < RETRIES:
            attempt += 1
            try:
                print(f"[{attempt}] snapshot_download {repo_id} -> {target}")
                path = snapshot_download_with_env(repo_id, target)
                print("Downloaded to:", path)
                break
            except Exception as e:
                print(f"snapshot_download failed (attempt {attempt}): {e}")
                traceback.print_exc()
                if attempt < RETRIES:
                    print(f"Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("Exceeded retries for", repo_id)
                    return 1

    # List small summary
    print("\nDownload completed. Directory summaries:")
    for p in (T5_DIR, MINILM_DIR):
        print("\n", p)
        try:
            for i, entry in enumerate(sorted(p.iterdir(), key=lambda x: x.name)):
                if i >= 40:
                    break
                print(f" - {entry.name}  {entry.stat().st_size}")
        except Exception as ex:
            print("Could not list", p, ":", ex)

    # Try local loads to verify
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from sentence_transformers import SentenceTransformer
        print("\nAttempting to load models locally (local_files_only=True)...")
        T5Tokenizer.from_pretrained(str(T5_DIR), local_files_only=True)
        T5ForConditionalGeneration.from_pretrained(str(T5_DIR), local_files_only=True)
        SentenceTransformer(str(MINILM_DIR))
        print("Local load checks passed.")
    except Exception:
        print("Local load check failed; traceback below:")
        traceback.print_exc()
        # still return success because downloads may have succeeded partially
    return 0

if __name__ == "__main__":
    sys.exit(main())