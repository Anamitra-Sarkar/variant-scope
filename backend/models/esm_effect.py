import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMEffectHead(nn.Module):
    def __init__(self, embedding_dim: int = 640, hidden_dim: int = 256):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.attn_proj = nn.Linear(embedding_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, embeddings, mutation_positions):
        batch_size, seq_len, embed_dim = embeddings.shape
        positions = mutation_positions.unsqueeze(1).float()
        indices = torch.arange(seq_len, device=embeddings.device).unsqueeze(0).float()
        distances = torch.abs(indices - positions)
        attention_weights = torch.exp(-self.gamma * distances)
        attention_weights = attention_weights.unsqueeze(-1)
        weighted = embeddings * attention_weights
        pooled = weighted.sum(dim=1)
        projected = self.attn_proj(pooled)
        logits = self.classifier(projected)
        return logits


class InductiveBiasPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.head = ESMEffectHead(
            embedding_dim=model.config.hidden_size
        ).to(device)
        self.head.eval()

    def predict(self, sequences, mutation_positions):
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.last_hidden_state

        positions = torch.tensor(mutation_positions, device=self.device)
        logits = self.head(embeddings, positions)
        probs = F.softmax(logits, dim=-1)
        return {
            "pathogenicity_score": probs[:, 1].cpu().tolist(),
            "benign_score": probs[:, 0].cpu().tolist(),
        }
