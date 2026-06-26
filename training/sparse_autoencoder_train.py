import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from sklearn.metrics import roc_auc_score
import argparse


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, sparsity_lambda: float = 1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        encoded = self.encoder(x)
        sparse = F.relu(encoded)
        reconstructed = self.decoder(sparse)
        recon_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = self.sparsity_lambda * sparse.mean(dim=0).abs().sum()
        return recon_loss + sparsity_loss, sparse


def extract_embeddings(model, tokenizer, sequences, device):
    model.eval()
    embeddings = []
    batch_size = 16

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(emb)

    return np.concatenate(embeddings, axis=0)


def train_sae(
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    sequences: list = None,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    output_path: str = "./sae.pt",
    hf_token: str = "",
    hf_repo: str = "Arko007/variantscope-models",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, tokenizer, sequences, device)
    input_dim = embeddings.shape[1]

    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)

    embeddings_tensor = torch.tensor(embeddings, device=device)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        sae.train()
        total_loss = 0.0

        for i in range(0, len(embeddings_tensor), 32):
            batch = embeddings_tensor[i:i + 32]
            loss, _ = sae(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(embeddings_tensor) // 32 + 1)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(sae.state_dict(), output_path)
            print(f"Saved best model with loss {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if hf_token:
        from huggingface_hub import HfApi, HfFolder
        api = HfApi()
        repo_id = f"{hf_repo}/sae"
        api.create_repo(repo_id, exist_ok=True, token=hf_token)
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo="sae.pt",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print(f"Uploaded SAE to {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    from dataset import create_synthetic_dataset
    seqs, _ = create_synthetic_dataset(n_samples=1000, seq_length=128)

    train_sae(
        sequences=seqs,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        hf_token=os.environ.get("HF_TOKEN", ""),
    )


if __name__ == "__main__":
    main()
