import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR    = "results"
TOP_PROT_TXT  = os.path.join(RESULT_DIR, "top_proteins_rf_dl.txt")
PROCESSED_CSV = os.path.join(RESULT_DIR, "xgb_processed_pairs.csv")
KEGG_GMT_PATH = "c2.cp.v2024.1.Hs.symbols.gmt"

def read_protein_set():
    if Path(TOP_PROT_TXT).exists():
        p = [line.strip() for line in open(TOP_PROT_TXT, "r", encoding="utf-8") if line.strip()]
        return sorted(set(p))
    df = pd.read_csv(PROCESSED_CSV)
    df = df[df["label"] == 1]
    return sorted(set(df["protein1"]).union(set(df["protein2"])))

def map_protein_to_symbol(ensp_list):
    mapping = Path("id_mapping.csv")
    if mapping.exists():
        df = pd.read_csv(mapping)
        pcol = next((c for c in df.columns if c.lower().startswith("protein")), "protein_id")
        gcol = next((c for c in df.columns if "symbol" in c.lower()), "gene_symbol")
        df = df[[pcol, gcol]].dropna().drop_duplicates()
        m  = dict(zip(df[pcol].astype(str), df[gcol].astype(str)))
        syms = sorted({m[x] for x in ensp_list if x in m and m[x]})
        print(f"[MAP] id_mapping.csv  {len(syms)}  symbol")
        return syms
    # 在线兜底
    try:
        from mygene import MyGeneInfo
        mg = MyGeneInfo()
        res = mg.querymany(list(set(ensp_list)), scopes="ensembl.protein", fields="symbol", species="human")
        syms = [r.get("symbol") for r in res if isinstance(r, dict) and r.get("symbol")]
        syms = sorted(set([s for s in syms if isinstance(s, str)]))
        print(f"[MAP] MyGeneInfo - {len(syms)} 个 symbol")
        return syms
    except Exception as e:
        print(f"[MAP] （{e}）， KEGG。")
        return []

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if not Path(KEGG_GMT_PATH).exists():
        print(f"[KEGG]  GMT：{KEGG_GMT_PATH}"); return

    proteins = read_protein_set()
    print(f"[INFO] -: {len(proteins)}")
    symbols = map_protein_to_symbol(proteins)
    if not symbols:
        print("[KEGG] - gene symbols。")
        return

    import gseapy as gp
    outdir = os.path.join(RESULT_DIR, "kegg_enrichment")
    enr = gp.enrichr(
        gene_list=symbols,
        gene_sets=KEGG_GMT_PATH,
        organism="Human",
        outdir=outdir,
        cutoff=0.5
    )
    res = enr.results
    os.makedirs(outdir, exist_ok=True)
    res_path = os.path.join(outdir, "kegg_results.csv")
    res.to_csv(res_path, index=False)
    print(f"[SAVE] KEGG - {res_path}")

    df = res.sort_values("Adjusted P-value").head(30)
    if len(df) == 0:
        print("[KEGG] "); return
    terms = df["Term"]; scores = -np.log10(df["Adjusted P-value"] + 1e-300)
    plt.figure(figsize=(10,6))
    plt.barh(terms, scores); plt.gca().invert_yaxis()
    plt.xlabel("-log10(Adjusted P-value)")
    plt.title("Top 30 KEGG-like Pathways (Human)")
    plt.tight_layout()
    png = os.path.join(outdir, "top_kegg.png")
    plt.savefig(png, dpi=300); plt.close()
    print(f"[SAVE] KEGG - {png}")

if __name__ == "__main__":
    main()
