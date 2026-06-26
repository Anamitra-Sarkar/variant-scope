import json
import requests
import pandas as pd
from typing import List, Dict, Any


class SomaticVariantAnnotator:
    def __init__(self, assembly: str = "hg38"):
        self.myvariant_url = "https://myvariant.info/v1/variant"
        self.civic_url = "https://civicdb.org/api/graphql"
        self.assembly = assembly

    def fetch_myvariant_batch(self, hgvs_ids: List[str]) -> List[Dict[str, Any]]:
        if not hgvs_ids:
            return []

        fields_query = (
            "dbnsfp.sift.score,dbnsfp.sift.pred,"
            "dbnsfp.polyphen2.hdiv.score,dbnsfp.polyphen2.hdiv.pred,"
            "cadd.phred,cadd.score"
        )

        payload = {
            "ids": ",".join(hgvs_ids),
            "fields": fields_query,
            "assembly": self.assembly,
        }

        headers = {"content-type": "application/x-www-form-urlencoded"}
        response = requests.post(self.myvariant_url, data=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"MyVariant API failed: {response.status_code}")
        return response.json()

    def fetch_civic_evidence(self, gene_symbol: str) -> Dict[str, Any]:
        graphql_query = """
        query ($geneSymbol: String!) {
          genes(name: $geneSymbol) {
            edges {
              node {
                name
                description
                variants {
                  name
                  singleNucleotideVariants {
                    representativeTranscript
                  }
                }
              }
            }
          }
        }
        """
        variables = {"geneSymbol": gene_symbol}
        payload = {"query": graphql_query, "variables": variables}
        headers = {"content-type": "application/json"}

        response = requests.post(self.civic_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"CIViC API failed: {response.status_code}")
        return response.json()

    def merge_and_save(self, hgvs_ids: List[str], gene_symbol: str, out_path: str):
        mv_data = self.fetch_myvariant_batch(hgvs_ids)
        civic_data = self.fetch_civic_evidence(gene_symbol)

        processed_variants = []
        for entry in mv_data:
            if "notfound" in entry:
                continue

            vid = entry.get("query", entry.get("_id", "unknown"))
            dbnsfp = entry.get("dbnsfp", {})
            cadd = entry.get("cadd", {})

            sift = dbnsfp.get("sift", {})
            sift_score = sift.get("score") if not isinstance(sift, list) else sift[0].get("score")
            sift_pred = sift.get("pred") if not isinstance(sift, list) else sift[0].get("pred")

            pp2 = dbnsfp.get("polyphen2", {}).get("hdiv", {})
            pp2_score = pp2.get("score") if not isinstance(pp2, list) else pp2[0].get("score")
            pp2_pred = pp2.get("pred") if not isinstance(pp2, list) else pp2[0].get("pred")

            processed_variants.append({
                "hgvs_id": vid,
                "sift_score": sift_score,
                "sift_pred": sift_pred,
                "polyphen_score": pp2_score,
                "polyphen_pred": pp2_pred,
                "cadd_phred": cadd.get("phred"),
            })

        df = pd.DataFrame(processed_variants)
        df.to_csv(out_path, index=False)
        return df
