import os
import json
from groq import Groq


class AgentDecisionEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY environment configuration.")
        self.client = Groq(api_key=api_key)
        self.system_prompt = (
            "You are a genomic hyperparameter tuning agent. Analyze validation "
            "metrics and output an optimized mutation context window strategy as JSON."
        )

    def deliberate_strategy(self, metrics: dict) -> dict:
        prompt = (
            f"Active Metrics:\n{json.dumps(metrics, indent=2)}\n"
            "Provide optimal training context size, tokenization strategy, "
            "and adapter targets in the output JSON."
        )

        chat_completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        return json.loads(chat_completion.choices[0].message.content)
