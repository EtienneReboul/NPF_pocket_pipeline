#!/usr/bin/env python3
"""
scripts/generate_msa_clustering_umap_dbscan_gmm_all_proteins_notebook.py
===========================================================================
Builds a single notebook, `notebook/msa_clustering/all_proteins_umap_dbscan_gmm.ipynb`,
that batch-processes every NPF protein under `data/msa/a3m/` the same way as
`all_proteins_tsne_dbscan_gmm.ipynb`, but embeds the MSA with UMAP instead of
t-SNE: DBSCAN is run first on the UMAP embedding purely to strip noise (the
`-1` label), then a Gaussian Mixture Model is swept on the remaining
(denoised) points only, with n_components ranging from `MIN_COMPONENTS` to
`n_dbscan_clusters + GMM_EXTRA_COMPONENTS` (the number of non-noise DBSCAN
clusters, plus a fixed headroom). The number of components is picked from the
knee of the BIC curve via KneeLocator (falling back to the global BIC minimum
if no knee is found), same convention as `scripts/gmm_conformation.py` and the
t-SNE-based notebooks. Sequences DBSCAN calls noise keep a `-1` label straight
through to the final clustering (they are never assigned to a GMM component).
Per-cluster consensus + `.a3m` subsets and uniform-control MSAs are written
out, plus a gradient-colored UMAP landscape plot and a cross-protein summary
table.

Usage:
    python scripts/generate_msa_clustering_umap_dbscan_gmm_all_proteins_notebook.py
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebook" / "msa_clustering" / "all_proteins_umap_dbscan_gmm.ipynb"

KERNELSPEC = {
    "display_name": "npf-notebook",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
    "version": "3.11.15",
}


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines, tags=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags} if tags else {},
        "outputs": [],
        "source": "\n".join(lines),
    }


def build_notebook() -> dict:
    cells = [
        md(
            "# MSA Clustering — All Proteins (UMAP + DBSCAN-denoised GMM)",
            "",
            "Batch version of `all_proteins_tsne_dbscan_gmm.ipynb`, with the embedding "
            "step swapped from t-SNE to **UMAP** (`UMAP_METRIC = \"jaccard\"`, matching the "
            "per-protein notebooks' UMAP configuration). For each protein: DBSCAN is run "
            "on the UMAP embedding first (eps auto-selected via a k-distance Kneedle "
            "knee) purely to strip noise, then a **Gaussian Mixture Model** is swept only "
            "on the DBSCAN-denoised points, with `n_components` ranging from "
            "`MIN_COMPONENTS` to `n_dbscan_clusters + GMM_EXTRA_COMPONENTS` (the number of "
            "non-noise DBSCAN clusters found for that protein, plus a fixed headroom). "
            "BIC is recorded at each step and the number of components is chosen from the "
            "**knee of the BIC curve** via `KneeLocator` (falling back to the global BIC "
            "minimum if no knee is found — the same convention used in "
            "`scripts/gmm_conformation.py`). Sequences DBSCAN calls noise keep their `-1` "
            "label straight through — they are never assigned to a GMM component. "
            "Per-cluster consensus sequences + `.a3m` subsets + uniform-control MSAs are "
            "written out, along with a gradient-colored UMAP landscape plot and a "
            "cross-protein summary table.",
            "",
            "Reads every `data/msa/a3m/*.a3m` file (override via the `PROTEINS` parameter "
            "below to run a subset). Outputs go to "
            "`results/msa_clustering/umap_dbscan_gmm/<protein>/`, its own folder so they "
            "don't collide with the t-SNE-based notebooks' outputs.",
        ),
        code(
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "from pathlib import Path",
            "from Bio import SeqIO",
            "from polyleven import levenshtein",
            "from kneed import KneeLocator",
            "from sklearn.cluster import DBSCAN",
            "from sklearn.mixture import GaussianMixture",
            "from sklearn.neighbors import NearestNeighbors",
            "import umap",
        ),
        code(
            'ROOT = Path("../..")  # notebook/msa_clustering/ is two levels below project root',
            'A3M_DIR = ROOT / "data" / "msa" / "a3m"',
            'RESULTS_DIR = ROOT / "results" / "msa_clustering" / "umap_dbscan_gmm"',
            "",
            "PROTEINS = None          # None = every *.a3m in A3M_DIR; or e.g. [\"NPF6.4_Q9LVE0\", ...] for a subset",
            "",
            "GAP_CUTOFF = 0.25        # drop sequences with > this fraction of gap columns",
            "MIN_SAMPLES = 10        # DBSCAN min_samples (AF-Cluster recommends >= 3, but 3 gave spurious clusters here)",
            "EPS_VAL = None          # set a float to use one fixed eps for every protein instead of per-protein Kneedle",
            "MIN_COMPONENTS = 1      # low end of the post-DBSCAN GMM sweep",
            "GMM_EXTRA_COMPONENTS = 20  # GMM sweep runs up to (n_dbscan_clusters + this) components",
            "MIN_SAMPLES_PER_COMPONENT = 5  # caps the sweep so components don't outnumber ~1/5th of the denoised sequences",
            "GMM_COVARIANCE_TYPE = \"full\"",
            "GMM_N_INIT = 5           # restarts per k for stability",
            'UMAP_METRIC = "jaccard"  # treats each one-hot row as a set of (position, residue) indicators —',
            "                         # matches/mismatches only, and unlike Euclidean it's bounded in [0, 1]",
            "                         # regardless of alignment length, so it needs no per-protein recalibration",
            "UMAP_N_NEIGHBORS = 15",
            "UMAP_MIN_DIST = 0.1",
            "N_CONTROLS = 10          # number of uniformly-sampled control MSAs per size",
            "RESAMPLE = False         # bootstrap-resample each MSA (with replacement) before clustering",
            "RANDOM_STATE = 0",
            'LANDSCAPE_CMAP = "turbo"  # continuous colormap for cluster IDs — no cutoff on number of clusters shown',
            "",
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)",
            "np.random.seed(RANDOM_STATE)",
            tags=["parameters"],
        ),
        md("## 1. Helper functions"),
        code(
            'AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"',
            "",
            "",
            "def load_fasta(path):",
            "    ids, seqs = [], []",
            '    for rec in SeqIO.parse(str(path), "fasta"):',
            "        ids.append(rec.id)",
            "        seqs.append(str(rec.seq))",
            "    return ids, seqs",
            "",
            "",
            "def strip_insertions(seqs):",
            '    """Keep only match-state columns (uppercase + gap), matching AF-Cluster preprocessing."""',
            "    return [''.join(c for c in s if c.isupper() or c == '-') for s in seqs]",
            "",
            "",
            "def encode_seqs(seqs, max_len, alphabet=AA_ALPHABET):",
            '    """One-hot encode equal-length sequences, flattened to (n_seqs, max_len * len(alphabet))."""',
            "    arr = np.array([list(s.ljust(max_len, '-')) for s in seqs])",
            "    onehot = np.stack([arr == c for c in alphabet], axis=-1).astype(np.float32)",
            "    return onehot.reshape(len(seqs), max_len * len(alphabet))",
            "",
            "",
            "def consensus_sequence(seqs):",
            '    """Per-column majority-vote consensus sequence."""',
            "    cols = np.array([list(s) for s in seqs])",
            "    consensus = []",
            "    for i in range(cols.shape[1]):",
            "        vals, counts = np.unique(cols[:, i], return_counts=True)",
            "        consensus.append(vals[np.argmax(counts)])",
            "    return ''.join(consensus)",
            "",
            "",
            "def avg_identity(seqs, ref, L):",
            '    """Mean fractional identity to `ref` via Levenshtein distance (AF-Cluster metric)."""',
            "    if len(seqs) == 0:",
            "        return np.nan",
            "    return np.mean([1 - levenshtein(s, ref) / L for s in seqs])",
            "",
            "",
            "def write_fasta(names, seqs, outfile):",
            "    with open(outfile, 'w') as f:",
            "        for name, seq in zip(names, seqs):",
            '            f.write(f">{name}\\n{seq}\\n")',
            "",
            "",
            "def select_eps_and_cluster(X, min_samples, eps_val, label, out_dir, protein):",
            '    """Kneedle k-distance eps selection (or a fixed eps_val) + DBSCAN fit on embedding X."""',
            "    if eps_val is None:",
            "        nn = NearestNeighbors(n_neighbors=min_samples).fit(X)",
            "        k_dist = np.sort(nn.kneighbors(X)[0][:, -1])",
            "",
            "        degree = min(7, max(1, len(k_dist) - 3))",
            '        kneedle = KneeLocator(np.arange(len(k_dist)), k_dist, curve="convex", direction="increasing",',
            '                              interp_method="polynomial", polynomial_degree=degree)',
            "        eps = float(k_dist[kneedle.knee]) if kneedle.knee is not None else float(np.median(k_dist))",
            "",
            "        fig, ax = plt.subplots(figsize=(5, 4))",
            '        ax.plot(np.arange(len(k_dist)), k_dist, color="steelblue", label="k-distance (sorted)")',
            '        ax.axhline(eps, color="red", ls="--", label=f"Kneedle eps={eps:.3f}")',
            '        ax.set_xlabel(f"points, sorted by distance to their {min_samples}-th nearest neighbor")',
            '        ax.set_ylabel("k-distance")',
            '        ax.set_title(f"{protein} — {label} k-distance")',
            "        ax.legend()",
            "        plt.tight_layout()",
            '        plt.savefig(out_dir / f"{protein}_{label}_eps_kneedle.pdf", bbox_inches="tight")',
            "        plt.close(fig)",
            "    else:",
            "        eps = eps_val",
            "",
            "    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_",
            "    return labels, eps",
            "",
            "",
            "def select_k_and_cluster_gmm(X, min_components, max_components, min_samples_per_component,",
            "                              n_init, covariance_type, random_state, label, out_dir, protein):",
            '    """BIC sweep n_components=min..max (Schwarz 1978), then pick k from the knee of the BIC',
            "    curve via KneeLocator (Satopaa et al. 2011), falling back to the global BIC minimum if no",
            '    knee is found — same convention as scripts/gmm_conformation.py\'s find_best_k. KneeLocator',
            "    is run with polynomial interpolation (rather than its default interp1d) so a single noisy",
            "    BIC value (e.g. a bad n_init restart) doesn't get re-interpolated as a spurious knee — the",
            '    polynomial degree is capped relative to len(ks) so the fit smooths rather than overfits."""',
            "    max_k = max(min_components, min(max_components, len(X) // min_samples_per_component))",
            "    ks = list(range(min_components, max_k + 1))",
            "",
            "    bics = []",
            "    for k in ks:",
            "        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,",
            "                              n_init=n_init, random_state=random_state, max_iter=300)",
            "        gmm.fit(X)",
            "        bics.append(gmm.bic(X))",
            "",
            "    argmin_k = ks[int(np.argmin(bics))]",
            "    knee_k = None",
            "    if len(ks) >= 3:",
            "        degree = min(7, max(1, len(ks) - 3))",
            "        try:",
            '            kneedle = KneeLocator(ks, bics, curve="convex", direction="decreasing",',
            '                                  interp_method="polynomial", polynomial_degree=degree)',
            "            knee_k = int(kneedle.knee) if kneedle.knee is not None else None",
            "        except Exception as e:",
            '            print(f"[{label}] WARNING: KneeLocator failed ({e}), falling back to BIC minimum")',
            "    best_k = knee_k if knee_k is not None else argmin_k",
            "",
            "    fig, ax = plt.subplots(figsize=(5, 4))",
            '    ax.plot(ks, bics, marker="o", color="steelblue", label="BIC")',
            '    ax.axvline(best_k, color="red", ls="--", label=f"selected k={best_k} (knee)")',
            "    if argmin_k != best_k:",
            '        ax.axvline(argmin_k, color="gray", ls=":", label=f"BIC minimum k={argmin_k}")',
            '    ax.set_xlabel("n_components")',
            '    ax.set_ylabel("BIC")',
            '    ax.set_title(f"{protein} — {label} BIC sweep")',
            "    ax.legend()",
            "    plt.tight_layout()",
            '    plt.savefig(out_dir / f"{protein}_{label}_bic.pdf", bbox_inches="tight")',
            "    plt.close(fig)",
            "",
            "    final_gmm = GaussianMixture(n_components=best_k, covariance_type=covariance_type,",
            "                                n_init=n_init, random_state=random_state, max_iter=300).fit(X)",
            "    labels = final_gmm.predict(X)",
            "    return labels, best_k, argmin_k",
            "",
            "",
            "def plot_landscape(x, y, df, query_row, label_col, title, outfile, cmap=LANDSCAPE_CMAP):",
            "    fig = plt.figure(figsize=(5, 5))",
            "",
            "    unclustered = df.loc[df[label_col] == -1]",
            "    if len(unclustered):",
            '        plt.scatter(unclustered[x], unclustered[y], color="lightgray", marker="x", label="noise (DBSCAN)")',
            "",
            "    clustered = df.loc[df[label_col] >= 0]",
            "    if len(clustered):",
            "        n_clust = clustered[label_col].nunique()",
            "        sc = plt.scatter(clustered[x], clustered[y], c=clustered[label_col], cmap=cmap,",
            "                         linewidth=0, s=25)",
            "        cbar = plt.colorbar(sc)",
            '        cbar.set_label(f"cluster ID ({n_clust} total)")',
            "",
            '    plt.scatter(query_row[x], query_row[y], color="red", marker="*", s=150, label="Ref Seq")',
            '    plt.legend(loc="upper left", bbox_to_anchor=(1.2, 1), frameon=False)',
            "    plt.xlabel(x)",
            "    plt.ylabel(y)",
            "    plt.title(title)",
            "    plt.tight_layout()",
            '    plt.savefig(outfile, bbox_inches="tight")',
            "    plt.close(fig)",
        ),
        md(
            "## 2. Per-protein pipeline",
            "",
            "One-hot encode → UMAP embed → Kneedle-selected DBSCAN (noise removal only) → "
            "BIC-swept, Kneedle-selected GMM on the denoised points → per-cluster "
            "consensus/`.a3m` + uniform controls + landscape plot. Returns a one-row "
            "summary dict used to build the cross-protein table in the last section.",
        ),
        code(
            "def process_protein(protein, a3m_path, out_dir):",
            "    out_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "    raw_ids, raw_seqs = load_fasta(a3m_path)",
            "    seqs = strip_insertions(raw_seqs)",
            '    df_all = pd.DataFrame({"SequenceName": raw_ids, "sequence": seqs})',
            "",
            "    query = df_all.iloc[:1].copy()",
            "    df = df_all.iloc[1:].copy()",
            "    if RESAMPLE:",
            "        df = df.sample(frac=1, random_state=RANDOM_STATE)",
            "",
            "    L = len(df.sequence.iloc[0])",
            '    df["frac_gaps"] = df.sequence.str.count("-") / L',
            "    df = df.loc[df.frac_gaps < GAP_CUTOFF].reset_index(drop=True)",
            "    n_after = len(df)",
            "",
            "    ohe = encode_seqs(df.sequence.tolist(), max_len=L)",
            "    umap_reducer = umap.UMAP(metric=UMAP_METRIC, n_neighbors=UMAP_N_NEIGHBORS,",
            "                              min_dist=UMAP_MIN_DIST, random_state=RANDOM_STATE)",
            "    umap_embedding = umap_reducer.fit_transform(ohe)",
            "    umap_query_embedding = umap_reducer.transform(encode_seqs(query.sequence.tolist(), max_len=L))",
            '    df["UMAP 1"], df["UMAP 2"] = umap_embedding[:, 0], umap_embedding[:, 1]',
            '    query["UMAP 1"], query["UMAP 2"] = umap_query_embedding[:, 0], umap_query_embedding[:, 1]',
            "",
            "    dbscan_labels, eps = select_eps_and_cluster(",
            '        df[["UMAP 1", "UMAP 2"]].values, MIN_SAMPLES, EPS_VAL, "dbscan", out_dir, protein',
            "    )",
            '    df["cluster_label_dbscan"] = dbscan_labels',
            "    n_dbscan_clusters = len(sorted(c for c in df.cluster_label_dbscan.unique() if c >= 0))",
            '    n_noise = int((df.cluster_label_dbscan == -1).sum())',
            "",
            "    df_clean = df.loc[df.cluster_label_dbscan != -1]",
            "    max_components = n_dbscan_clusters + GMM_EXTRA_COMPONENTS",
            "    if len(df_clean) == 0:",
            '        print(f"[{protein}] WARNING: DBSCAN labeled every sequence as noise; skipping GMM sweep")',
            "        gmm_labels, best_k, argmin_k = np.array([], dtype=int), 0, 0",
            "    else:",
            "        gmm_labels, best_k, argmin_k = select_k_and_cluster_gmm(",
            '            df_clean[["UMAP 1", "UMAP 2"]].values, MIN_COMPONENTS, max_components,',
            "            MIN_SAMPLES_PER_COMPONENT, GMM_N_INIT, GMM_COVARIANCE_TYPE, RANDOM_STATE,",
            '            "dbscan_gmm", out_dir, protein,',
            "        )",
            '    df["cluster_label_dbscan_gmm"] = -1',
            '    df.loc[df_clean.index, "cluster_label_dbscan_gmm"] = gmm_labels',
            "",
            '    avg_id_to_query = avg_identity(df.sequence.tolist(), query.sequence.iloc[0], L)',
            '    id_noise = avg_identity(df.loc[df.cluster_label_dbscan_gmm == -1, "sequence"].tolist(),',
            "                            query.sequence.iloc[0], L)",
            '    id_clustered = avg_identity(df.loc[df.cluster_label_dbscan_gmm != -1, "sequence"].tolist(),',
            "                                query.sequence.iloc[0], L)",
            "",
            "    clusters = sorted(c for c in df.cluster_label_dbscan_gmm.unique() if c >= 0)",
            "    records = []",
            "    for clust in clusters:",
            "        tmp = df.loc[df.cluster_label_dbscan_gmm == clust]",
            "        cs = consensus_sequence(tmp.sequence.tolist())",
            "        avg_dist_to_cs = avg_identity(tmp.sequence.tolist(), cs, L)",
            "        avg_dist_to_query = avg_identity(tmp.sequence.tolist(), query.sequence.iloc[0], L)",
            "        records.append({",
            '            "cluster_ind": clust,',
            '            "consensusSeq": cs,',
            '            "avg_lev_dist": round(avg_dist_to_cs, 3),',
            '            "avg_dist_to_query": round(avg_dist_to_query, 3),',
            '            "size": len(tmp),',
            "        })",
            "        cluster_with_query = pd.concat([query, tmp], axis=0)",
            "        write_fasta(cluster_with_query.SequenceName.tolist(), cluster_with_query.sequence.tolist(),",
            '                    out_dir / f"{protein}_dbscan_gmm_{clust:03d}.a3m")',
            "",
            "    metadata_df = pd.DataFrame.from_records(",
            "        records,",
            '        columns=["cluster_ind", "consensusSeq", "avg_lev_dist", "avg_dist_to_query", "size"],',
            "    )",
            "    if len(metadata_df):",
            '        metadata_df = metadata_df.sort_values("size", ascending=False)',
            '    metadata_df.to_csv(out_dir / f"{protein}_dbscan_gmm_cluster_metadata.tsv", index=False, sep="\\t")',
            "",
            "    for i in range(N_CONTROLS):",
            "        tmp = df.sample(n=min(10, len(df)), random_state=RANDOM_STATE + i)",
            "        tmp = pd.concat([query, tmp], axis=0)",
            '        write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), out_dir / f"{protein}_U10-{i:03d}.a3m")',
            "    if len(df) > 100:",
            "        for i in range(N_CONTROLS):",
            "            tmp = df.sample(n=100, random_state=RANDOM_STATE + i)",
            "            tmp = pd.concat([query, tmp], axis=0)",
            '            write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), out_dir / f"{protein}_U100-{i:03d}.a3m")',
            "",
            '    df.to_csv(out_dir / f"{protein}_dbscan_gmm_clustering_assignments.tsv", index=False, sep="\\t")',
            "",
            '    plot_landscape("UMAP 1", "UMAP 2", df, query.iloc[0], "cluster_label_dbscan_gmm",',
            '                   f"{protein} — DBSCAN-denoised GMM (k={best_k})",',
            '                   out_dir / f"{protein}_UMAP_DBSCAN_GMM.pdf")',
            "",
            "    return {",
            '        "protein": protein,',
            '        "n_seqs": n_after,',
            '        "L": L,',
            '        "eps": round(eps, 3),',
            '        "n_dbscan_clusters": n_dbscan_clusters,',
            '        "n_noise": n_noise,',
            '        "pct_noise": round(n_noise / n_after, 4) if n_after else float("nan"),',
            '        "gmm_max_components_tried": max_components,',
            '        "n_components_knee": best_k,',
            '        "n_components_bic_min": argmin_k,',
            '        "avg_identity_to_query": round(avg_id_to_query, 3),',
            '        "avg_identity_clustered": round(id_clustered, 3) if pd.notna(id_clustered) else id_clustered,',
            '        "avg_identity_noise": round(id_noise, 3) if pd.notna(id_noise) else id_noise,',
            "    }",
        ),
        md(
            "## 3. Run over all proteins",
            "",
            "Processes each protein in turn; a failure on one protein is logged and "
            "skipped rather than aborting the whole batch.",
        ),
        code(
            "if PROTEINS is None:",
            "    PROTEINS = sorted(p.stem for p in A3M_DIR.glob('*.a3m'))",
            'print(f"Processing {len(PROTEINS)} protein(s)")',
            "",
            "summaries = []",
            "for protein in PROTEINS:",
            '    print(f"=== {protein} ===")',
            "    try:",
            "        summary = process_protein(protein, A3M_DIR / f'{protein}.a3m', RESULTS_DIR / protein)",
            '        print(f"  DBSCAN: eps={summary[\'eps\']:.3f} -> {summary[\'n_dbscan_clusters\']} clusters, "',
            '              f"{summary[\'n_noise\']}/{summary[\'n_seqs\']} noise ({summary[\'pct_noise\']:.1%})")',
            '        print(f"  GMM: n_components knee={summary[\'n_components_knee\']}, "',
            '              f"BIC-min={summary[\'n_components_bic_min\']} "',
            '              f"(swept up to {summary[\'gmm_max_components_tried\']})")',
            "    except Exception as e:",
            '        print(f"  FAILED: {e}")',
            '        summary = {"protein": protein, "error": str(e)}',
            "    summaries.append(summary)",
        ),
        md("## 4. Cross-protein summary"),
        code(
            "summary_df = pd.DataFrame.from_records(summaries)",
            'summary_outfile = RESULTS_DIR / "all_proteins_umap_dbscan_gmm_summary.tsv"',
            'summary_df.to_csv(summary_outfile, index=False, sep="\\t")',
            'print(f"Wrote summary for {len(summary_df)} protein(s) to {summary_outfile}")',
            "summary_df",
        ),
    ]

    for cell in cells:
        cell["source"] = cell["source"].splitlines(keepends=True)
        cell["id"] = uuid.uuid4().hex[:8]
        if cell["cell_type"] == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)

    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNELSPEC, "language_info": LANGUAGE_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
