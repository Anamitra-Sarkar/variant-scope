import torch
import torch.nn.functional as F


class DNABERT2Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, sequences):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            pooled = embeddings[:, 0, :]

        classifier = getattr(self.model, "classifier", None)
        if classifier is not None:
            logits = classifier(pooled)
        else:
            logits = torch.nn.Linear(
                pooled.shape[-1], 2, device=self.device
            )(pooled)

        probs = F.softmax(logits, dim=-1)
        return {
            "pathogenicity_score": probs[:, 1].cpu().tolist(),
            "benign_score": probs[:, 0].cpu().tolist(),
            "embedding": pooled.cpu().numpy(),
        }
