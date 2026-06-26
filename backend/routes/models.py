from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["Models"])

AVAILABLE_MODELS = [
    {
        "id": "esm2",
        "name": "ESM-2 (150M)",
        "description": "Evolutionary Scale Modeling v2 - Protein language model for variant effect prediction",
        "parameters": "150M",
        "type": "protein",
        "architecture": "Bidirectional Transformer Encoder",
        "attention": "Rotary Positional Embeddings (RoPE)",
        "finetuning_strategies": ["LoRA", "BitFit", "Inductive Head", "Layer Unfreezing"],
    },
    {
        "id": "dnabert2",
        "name": "DNABERT-2 (117M)",
        "description": "DNA bidirectional encoder with BPE tokenization for genomic variant analysis",
        "parameters": "117M",
        "type": "genomic",
        "architecture": "Transformer Encoder with GLU MLP",
        "attention": "Attention with Linear Biases (ALiBi)",
        "tokenization": "Byte Pair Encoding (4096 subwords)",
        "finetuning_strategies": ["LoRA", "BitFit", "Layer Unfreezing"],
    },
    {
        "id": "hybrid",
        "name": "Hybrid ESM-2 + DNABERT-2",
        "description": "Joint ensemble with PLM-SAE disentanglement and cross-fitness routing",
        "parameters": "267M",
        "type": "hybrid",
        "components": ["ESM-2-150M", "DNABERT-2-117M", "PLM-SAE", "Cross-Fitness Router"],
    },
]


@router.get("")
async def list_models():
    return {"models": AVAILABLE_MODELS}


@router.get("/{model_id}")
async def get_model(model_id: str):
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            return m
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
