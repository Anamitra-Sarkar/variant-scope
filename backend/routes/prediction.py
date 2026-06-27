from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import time
import math
import torch

router = APIRouter(prefix="/api", tags=["Prediction"])


class VariantRequest(BaseModel):
    sequence: str
    model_type: str = "esm2"
    position: Optional[int] = None
    ref_aa: Optional[str] = None
    alt_aa: Optional[str] = None


class BatchVariantRequest(BaseModel):
    variants: List[VariantRequest]


class VariantResponse(BaseModel):
    variant: str
    model_type: str
    pathogenicity_score: float
    benign_score: float
    confidence: float
    prediction: str
    inference_time_ms: float


def get_model_loader():
    from app import model_loader
    return model_loader


def get_sae():
    from app import sae_model
    return sae_model


def _boost_score(score: float) -> float:
    if score <= 0.0 or score >= 1.0:
        return score
    score = max(1e-7, min(1 - 1e-7, score))
    logit = math.log(score / (1 - score))
    boosted = 1.0 / (1.0 + math.exp(-logit * 10))
    return round(boosted, 4)


@router.post("/predict", response_model=VariantResponse)
async def predict_variant(
    request: VariantRequest,
    authorization: str = Header(None),
):
    from app import model_loader

    start = time.time()

    if request.model_type == "esm2":
        model, tokenizer = model_loader.load_esm()
        if model is None:
            raise HTTPException(500, "ESM-2 model not loaded")

        from models.esm_effect import InductiveBiasPredictor
        predictor = InductiveBiasPredictor(model, tokenizer, model_loader.device)

        pos = request.position or len(request.sequence) // 2
        result = predictor.predict([request.sequence], [pos])

    elif request.model_type == "dnabert2":
        model, tokenizer = model_loader.load_dnabert()
        if model is None:
            raise HTTPException(500, "DNABERT-2 model not loaded")

        from models.dnabert_predictor import DNABERT2Predictor
        predictor = DNABERT2Predictor(model, tokenizer, model_loader.device)
        result = predictor.predict([request.sequence])

    elif request.model_type == "hybrid":
        esm_model, esm_tokenizer = model_loader.load_esm()
        dna_model, dna_tokenizer = model_loader.load_dnabert()
        if esm_model is None or dna_model is None:
            raise HTTPException(500, "Models not loaded")

        from models.esm_effect import InductiveBiasPredictor
        from models.dnabert_predictor import DNABERT2Predictor

        esm_pred = InductiveBiasPredictor(esm_model, esm_tokenizer, model_loader.device)
        dna_pred = DNABERT2Predictor(dna_model, dna_tokenizer, model_loader.device)

        pos = request.position or len(request.sequence) // 2
        esm_result = esm_pred.predict([request.sequence], [pos])
        dna_result = dna_pred.predict([request.sequence])

        esm_score = esm_result["pathogenicity_score"][0]
        dna_score = dna_result["pathogenicity_score"][0]
        result = {
            "pathogenicity_score": [0.6 * esm_score + 0.4 * dna_score],
            "benign_score": [1.0 - (0.6 * esm_score + 0.4 * dna_score)],
        }
    else:
        raise HTTPException(400, f"Unknown model_type: {request.model_type}")

    elapsed = (time.time() - start) * 1000
    raw_score = result["pathogenicity_score"][0]
    score = _boost_score(raw_score)

    user_id = None
    if authorization:
        from firebase_auth import verify_token
        try:
            decoded = await verify_token(authorization)
            user_id = decoded.get("uid")
        except HTTPException:
            pass

    if user_id:
        from database import save_prediction
        import datetime
        save_prediction(user_id, {
            "variant": request.sequence[:20] + "..." if len(request.sequence) > 20 else request.sequence,
            "model_type": request.model_type,
            "pathogenicity_score": score,
            "benign_score": round(1.0 - score, 4),
            "confidence": round(abs(score - 0.5) * 2, 4),
            "prediction": "pathogenic" if score > 0.5 else "benign",
            "inference_time_ms": round(elapsed, 2),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })

    return VariantResponse(
        variant=request.sequence[:20] + "..." if len(request.sequence) > 20 else request.sequence,
        model_type=request.model_type,
        pathogenicity_score=score,
        benign_score=round(1.0 - score, 4),
        confidence=round(abs(score - 0.5) * 2, 4),
        prediction="pathogenic" if score > 0.5 else "benign",
        inference_time_ms=round(elapsed, 2),
    )


@router.post("/predict/batch")
async def predict_batch(request: BatchVariantRequest):
    results = []
    for variant in request.variants:
        single_request = VariantRequest(**variant.model_dump())
        result = await predict_variant(single_request)
        results.append(result)
    return {"results": results, "count": len(results)}
