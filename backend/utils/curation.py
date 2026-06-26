import pandas as pd
from typing import Optional


def filter_clinvar_variants(
    input_path: str,
    output_path: str,
    max_gene_length: int = 100000,
    min_review_stars: int = 1,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep="\t", compression="gzip", low_memory=False)

    valid_types = ["missense_variant", "nonsense_variant", "synonymous_variant"]
    df = df[df["Type"].isin(valid_types)]

    valid_significance = ["Benign", "Likely_benign", "Pathogenic", "Likely_pathogenic"]
    df = df[df["ClinicalSignificance"].isin(valid_significance)]

    df = df[df["ReviewStatus"].str.contains("star", na=False)]

    gene_counts = df["GeneSymbol"].value_counts()
    exon_genes = gene_counts[gene_counts > 5000].index.tolist()
    df = df[~df["GeneSymbol"].isin(exon_genes)]

    df.to_csv(output_path, index=False)
    return df


def resolve_transcripts(df: pd.DataFrame, gene_col: str = "GeneSymbol") -> pd.DataFrame:
    transcript_counts = df.groupby([gene_col, "Name"]).size().reset_index(name="count")
    best_transcripts = transcript_counts.loc[
        transcript_counts.groupby(gene_col)["count"].idxmax()
    ]
    return df.merge(
        best_transcripts[[gene_col, "Name"]], on=gene_col, suffixes=("", "_best")
    )
