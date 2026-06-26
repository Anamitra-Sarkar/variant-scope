from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import json

router = APIRouter(prefix="/api/agent", tags=["Agent"])


class MetricsRequest(BaseModel):
    validation_loss: float
    auroc: float
    context_window_size: int
    tokenization: str
    low_resource_samples: int


class StrategyResponse(BaseModel):
    recommended_window_size: int
    tokenization_strategy: str
    prompt_framing: str
    adapter_targets: str
    learning_rate: float
    reasoning: str


@router.post("/tune", response_model=StrategyResponse)
async def tune_hyperparameters(metrics: MetricsRequest):
    groq_api_key = __import__("os").environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        context = metrics.context_window_size
        if metrics.auroc < 0.75:
            new_window = min(context * 2, 1024)
            framing = "semantic"
        elif metrics.auroc < 0.80:
            new_window = context
            framing = "physiological"
        else:
            new_window = context
            framing = "symmetric"

        return StrategyResponse(
            recommended_window_size=new_window,
            tokenization_strategy="BPE-600" if metrics.auroc < 0.78 else metrics.tokenization,
            prompt_framing=framing,
            adapter_targets="query,value",
            learning_rate=2e-4 if metrics.low_resource_samples < 500 else 5e-5,
            reasoning="Fallback heuristic (Groq API not configured)",
        )

    from groq import Groq
    client = Groq(api_key=groq_api_key)

    system_prompt = (
        "You are a genomic hyperparameter tuning agent. Analyze validation "
        "metrics and output an optimized mutation context window strategy as JSON."
    )

    user_prompt = (
        f"Metrics:\n{json.dumps(metrics.model_dump(), indent=2)}\n\n"
        "Output JSON with: recommended_window_size, tokenization_strategy (BPE-600 or Overlapping k-mer), "
        "prompt_framing (symmetric/physiological/semantic), adapter_targets, learning_rate, reasoning"
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    result = json.loads(chat_completion.choices[0].message.content)
    return StrategyResponse(**result)
