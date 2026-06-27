import math
import torch
import torch.nn.functional as F


class ZeroShotESMPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._token_to_id = tokenizer.get_vocab()

    def predict(self, sequences, mutation_positions, ref_aas=None, alt_aas=None):
        scores = []
        for i, seq in enumerate(sequences):
            pos = mutation_positions[i]
            ref = ref_aas[i] if ref_aas else seq[pos]
            alt = alt_aas[i] if alt_aas else None
            scores.append(self._score_mutation(seq, pos, ref, alt))
        return {
            "pathogenicity_score": scores,
            "benign_score": [round(1.0 - s, 4) for s in scores],
            "_conservation": alt_aas is None or all(a is None for a in alt_aas),
        }

    def _aa_token_id(self, aa):
        ids = self.tokenizer.encode(aa, add_special_tokens=False)
        if ids:
            return ids[0]
        return self._token_to_id.get(aa)

    def _score_mutation(self, sequence, position, ref_aa, alt_aa):
        inputs = self.tokenizer(
            [sequence], return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        token_pos = position + 1
        masked_ids = inputs["input_ids"].clone()
        masked_ids[0, token_pos] = self.tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(input_ids=masked_ids, attention_mask=inputs["attention_mask"])
            logits = outputs.logits[0, token_pos, :]
            probs = F.softmax(logits, dim=-1)

        ref_token_id = self._aa_token_id(ref_aa)
        if ref_token_id is None:
            return 0.5
        p_ref = max(probs[ref_token_id].item(), 1e-10)

        if alt_aa is None or alt_aa == ref_aa:
            return round(p_ref, 4)

        alt_token_id = self._aa_token_id(alt_aa)
        if alt_token_id is None:
            return 0.5
        p_alt = max(probs[alt_token_id].item(), 1e-10)

        llr = math.log(p_alt / p_ref)
        score = 1.0 / (1.0 + math.exp(llr))
        return round(score, 4)
