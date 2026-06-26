import argparse
import numpy as np
from train_esm_lora import ESM2Trainer
from train_dnabert_lora import DNABERT2Trainer
from sparse_autoencoder_train import train_sae
from dataset import create_synthetic_dataset, split_dataset
import os


def train_hybrid(
    hf_token: str = "",
    hf_repo: str = "Arko007/variantscope-models",
    output_dir: str = "./hybrid_output",
):
    seqs, labels = create_synthetic_dataset(n_samples=500, seq_length=128)
    (train_s, train_l), (val_s, val_l), (test_s, test_l) = split_dataset(seqs, labels)

    print("=" * 60)
    print("Training ESM-2 with LoRA")
    print("=" * 60)
    esm_trainer = ESM2Trainer(
        model_name="facebook/esm2_t30_150M_UR50D",
        use_lora=True,
        hf_token=hf_token,
        hf_repo=hf_repo,
    )
    esm_auroc = esm_trainer.train(
        train_seqs=train_s, train_labels=train_l,
        val_seqs=val_s, val_labels=val_l,
        output_dir=f"{output_dir}/esm2",
        learning_rate=2e-4,
        num_epochs=20,
        patience=5,
    )

    print("=" * 60)
    print("Training DNABERT-2 with LoRA")
    print("=" * 60)
    dna_trainer = DNABERT2Trainer(
        hf_token=hf_token,
        hf_repo=hf_repo,
    )
    dna_auroc = dna_trainer.train(
        train_seqs=train_s, train_labels=train_l,
        val_seqs=val_s, val_labels=val_l,
        output_dir=f"{output_dir}/dnabert2",
        learning_rate=2e-4,
        num_epochs=20,
        patience=5,
    )

    print("=" * 60)
    print("Training Sparse Autoencoder")
    print("=" * 60)
    train_sae(
        sequences=train_s + val_s,
        hidden_dim=256,
        epochs=50,
        patience=5,
        output_path=f"{output_dir}/sae.pt",
        hf_token=hf_token,
        hf_repo=hf_repo,
    )

    print("=" * 60)
    print(f"Results:")
    print(f"  ESM-2 AUROC:    {esm_auroc:.4f}")
    print(f"  DNABERT-2 AUROC: {dna_auroc:.4f}")
    print(f"  Best saved to HuggingFace: {hf_repo}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./hybrid_output")
    args = parser.parse_args()

    train_hybrid(
        hf_token=os.environ.get("HF_TOKEN", ""),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
