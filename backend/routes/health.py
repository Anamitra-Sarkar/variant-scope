from fastapi import APIRouter
import torch

router = APIRouter(tags=["Health"])


@router.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "version": "1.0.0",
        "service": "VariantScope API",
    }
