import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split


def load_clinvar_clinically_filtered(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)

    valid_types = ["missense_variant", "nonsense_variant"]
    df = df[df["Type"].isin(valid_types)]

    valid_sig = ["Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"]
    df = df[df["ClinicalSignificance"].isin(valid_sig)]

    if "ReviewStatus" in df.columns:
        df = df[df["ReviewStatus"].str.contains("star", na=False)]

    exclude_genes = ["TTN", "MUC16", "MUC17", "MUC4", "NEB", "OBSCN"]
    df = df[~df["GeneSymbol"].isin(exclude_genes)]

    return df


def extract_sequences_from_variants(
    df: pd.DataFrame,
    genome_fasta: str,
    transcript_gtf: str,
    flanking_bp: int = 512,
) -> pd.DataFrame:
    from Bio import SeqIO

    genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

    records = []
    for _, row in df.iterrows():
        chrom = row.get("Chromosome", "")
        pos = int(row.get("Start", 0))
        ref = row.get("ReferenceAllele", "")
        alt = row.get("AlternateAllele", "")

        if not chrom or not pos:
            continue

        chrom = chrom.replace("chr", "")
        seq_id = f"chr{chrom}"
        if seq_id not in genome:
            continue

        start = max(0, pos - flanking_bp)
        end = pos + len(ref) + flanking_bp
        ref_seq = str(genome[seq_id].seq[start:end]).upper()
        alt_seq = ref_seq[:flanking_bp] + alt + ref_seq[flanking_bp + len(ref):]

        label = 1 if "Pathogenic" in str(row.get("ClinicalSignificance", "")) else 0

        records.append({
            "chrom": chrom,
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "ref_seq": ref_seq,
            "alt_seq": alt_seq,
            "label": label,
            "gene": row.get("GeneSymbol", ""),
        })

    return pd.DataFrame(records)


def create_synthetic_dataset(
    n_samples: int = 500,
    seq_length: int = 128,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    np.random.seed(seed)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    sequences = []
    labels = []

    for _ in range(n_samples):
        seq = "".join(np.random.choice(amino_acids, size=seq_length))
        label = 1 if np.random.random() > 0.7 else 0
        sequences.append(seq)
        labels.append(label)

    return sequences, labels


def split_dataset(
    sequences: List[str],
    labels: List[int],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
):
    seq_train, seq_temp, lab_train, lab_temp = train_test_split(
        sequences, labels, test_size=test_size + val_size, random_state=seed
    )
    val_adjust = val_size / (test_size + val_size)
    seq_val, seq_test, lab_val, lab_test = train_test_split(
        seq_temp, lab_temp, test_size=val_adjust, random_state=seed
    )
    return (seq_train, lab_train), (seq_val, lab_val), (seq_test, lab_test)
