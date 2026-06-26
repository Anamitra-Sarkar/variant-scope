import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def run_ablation_study(
    sequences: list,
    labels: list,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    strategies: list = None,
):
    if strategies is None:
        strategies = ["symmetric_64", "symmetric_128", "symmetric_512",
                       "physiological_128", "physiological_512",
                       "semantic_128", "semantic_512"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, trust_remote_code=True
    ).to(device)
    model.eval()

    results = []

    for strategy in strategies:
        parts = strategy.split("_")
        framing = parts[0]
        window = int(parts[1]) if len(parts) > 1 else 128

        predictions = []
        for seq in sequences[:100]:
            if framing == "semantic":
                framed = f"[GENE] sample [MUT] {seq[:window]} [WT] {seq[-window:]}"
            elif framing == "physiological":
                framed = seq[:window]
            else:
                mid = len(seq) // 2
                half = window // 2
                framed = seq[max(0, mid - half):mid + half]

            inputs = tokenizer(framed, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions.append(probs[0, 1].item())

        auroc = roc_auc_score(labels[:100], predictions) if len(np.unique(labels[:100])) > 1 else 0.0
        entropy = -np.mean([p * np.log(p + 1e-10) + (1 - p) * np.log(1 - p + 1e-10) for p in predictions])

        results.append({
            "strategy": strategy,
            "framing": framing,
            "window": window,
            "auroc": round(auroc, 4),
            "entropy": round(entropy, 4),
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("ablation_results.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    args = parser.parse_args()

    from dataset import create_synthetic_dataset
    seqs, labels = create_synthetic_dataset(n_samples=200, seq_length=512)
    run_ablation_study(seqs, labels, model_name=args.model)


if __name__ == "__main__":
    main()
