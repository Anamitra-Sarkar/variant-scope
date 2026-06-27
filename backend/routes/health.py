from fastapi import APIRouter
import torch
import subprocess

router = APIRouter(tags=["Health"])

_COMMIT_HASH = ""
try:
    _COMMIT_HASH = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, timeout=5
    ).stdout.strip()
except Exception:
    _COMMIT_HASH = "unknown"


@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "1.0.0",
        "service": "VariantScope API",
        "commit": _COMMIT_HASH,
    }
