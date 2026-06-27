import os
import torch
import warnings
from huggingface_hub import hf_hub_download, snapshot_download

warnings.filterwarnings("ignore", category=FutureWarning)

from config import HF_MODEL_REPO, HF_TOKEN, MODEL_CACHE_DIR


class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.esm_model = None
        self.esm_tokenizer = None
        self.dnabert_model = None
        self.dnabert_tokenizer = None
        self.sae_model = None
        self.classifier = None

    def load_esm(self, model_name: str = "facebook/esm2_t30_150M_UR50D"):
        if self.esm_model is not None:
            return self.esm_model, self.esm_tokenizer

        from transformers import AutoTokenizer, AutoModel

        print(f"Loading ESM-2 model: {model_name}")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.esm_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self.esm_model.eval()
        return self.esm_model, self.esm_tokenizer

    def load_dnabert(self, model_name: str = "zhihan1996/DNABERT-2-117M"):
        if self.dnabert_model is not None:
            return self.dnabert_model, self.dnabert_tokenizer

        from transformers import AutoTokenizer, AutoModel

        print(f"Loading DNABERT-2 model: {model_name}")
        self.dnabert_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.dnabert_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self.dnabert_model.eval()
        return self.dnabert_model, self.dnabert_tokenizer

    def load_finetuned(self, repo_id: str = None):
        if repo_id is None:
            repo_id = HF_MODEL_REPO

        cache_dir = os.path.join(MODEL_CACHE_DIR, "finetuned")
        os.makedirs(cache_dir, exist_ok=True)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            model_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                token=HF_TOKEN or None,
                ignore_patterns=["*.pt", "*.bin", "*.msgpack"],
            )
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                model_path, trust_remote_code=True
            ).to(self.device)
            self.classifier.eval()
            return self.classifier
        except Exception as e:
            print(f"Could not load finetuned model: {e}")
            return None

    def load_sae(self, repo_id: str = None):
        if repo_id is None:
            repo_id = f"{HF_MODEL_REPO}/sae"

        cache_dir = os.path.join(MODEL_CACHE_DIR, "sae")
        os.makedirs(cache_dir, exist_ok=True)

        try:
            from models.sparse_autoencoder import SparseAutoencoder

            sae_path = hf_hub_download(
                repo_id=repo_id,
                filename="sae.pt",
                cache_dir=cache_dir,
                token=HF_TOKEN or None,
            )
            state = torch.load(sae_path, map_location=self.device)
            input_dim = state["encoder.weight"].shape[1]
            hidden_dim = state["encoder.weight"].shape[0]
            self.sae_model = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
            self.sae_model.load_state_dict(state)
            self.sae_model.to(self.device)
            self.sae_model.eval()
            return self.sae_model
        except Exception as e:
            print(f"Could not load SAE model: {e}")
            return None

    def unload(self):
        self.esm_model = None
        self.esm_tokenizer = None
        self.dnabert_model = None
        self.dnabert_tokenizer = None
        self.sae_model = None
        self.classifier = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
