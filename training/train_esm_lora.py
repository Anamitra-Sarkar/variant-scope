import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import argparse


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=1)

    auroc = roc_auc_score(labels, probs[:, 1]) if len(np.unique(labels)) > 1 else 0.0
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {
        "auroc": auroc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class ESM2Trainer:
    def __init__(
        self,
        model_name: str = "facebook/esm2_t30_150M_UR50D",
        num_labels: int = 2,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        hf_token: str = "",
        hf_repo: str = "Arko007/variantscope-models",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora
        self.hf_token = hf_token
        self.hf_repo = hf_repo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        print(f"Loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True,
        ).to(self.device)

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        target_modules = ["query", "value", "key", "dense"]
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def train(
        self,
        train_seqs: list,
        train_labels: list,
        val_seqs: list,
        val_labels: list,
        output_dir: str = "./esm2_finetuned",
        learning_rate: float = 2e-4,
        per_device_batch_size: int = 8,
        num_epochs: int = 20,
        patience: int = 5,
    ):
        train_enc = self.tokenizer(train_seqs, truncation=True, padding=True, max_length=512)
        val_enc = self.tokenizer(val_seqs, truncation=True, padding=True, max_length=512)

        train_enc["label"] = train_labels
        val_enc["label"] = val_labels

        train_dataset = Dataset.from_dict(train_enc)
        val_dataset = Dataset.from_dict(val_enc)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size * 2,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            save_total_limit=3,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )

        trainer.train()

        best_auroc = trainer.state.log_history[-1].get("eval_auroc", 0.0)
        print(f"Best validation AUROC: {best_auroc:.4f}")

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        if self.hf_token and best_auroc > 0.7:
            self._upload_to_hf(output_dir)

        return best_auroc

    def _upload_to_hf(self, model_dir: str):
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{self.hf_repo}/esm2-lora"
        api.create_repo(repo_id, exist_ok=True, token=self.hf_token)
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            token=self.hf_token,
        )
        print(f"Uploaded model to {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", default="./esm2_finetuned")
    args = parser.parse_args()

    from dataset import create_synthetic_dataset, split_dataset

    print("Creating synthetic dataset for testing")
    seqs, labels = create_synthetic_dataset(n_samples=500, seq_length=128)
    (train_s, train_l), (val_s, val_l), (test_s, test_l) = split_dataset(seqs, labels)

    trainer = ESM2Trainer(
        model_name=args.model,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        hf_token=os.environ.get("HF_TOKEN", ""),
    )

    best_auroc = trainer.train(
        train_seqs=train_s,
        train_labels=train_l,
        val_seqs=val_s,
        val_labels=val_l,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    print(f"Training complete. Best AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
