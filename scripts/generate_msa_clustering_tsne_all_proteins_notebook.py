#!/usr/bin/env python3
"""
scripts/generate_msa_clustering_tsne_all_proteins_notebook.py
===============================================================
Builds a single notebook, `notebook/msa_clustering/all_proteins_tsne_dbscan.ipynb`,
that batch-processes every NPF protein under `data/msa/a3m/` with just the
t-SNE + DBSCAN half of the per-protein MSA-clustering workflow (t-SNE gave
more compact, reliable separation than PCA/UMAP in practice, so PCA and UMAP
are skipped here entirely for speed across all proteins). For each protein:
one-hot encode the MSA, t-SNE embed it, auto-select DBSCAN's eps via a
k-distance Kneedle knee, cluster, write per-cluster consensus + .a3m subsets
and uniform-control MSAs, plot the t-SNE landscape (gradient-colored, no
10-cluster cutoff), and record summary stats. A cross-protein summary table
is written at the end.

Usage:
    python scripts/generate_msa_clustering_tsne_all_proteins_notebook.py
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebook" / "msa_clustering" / "all_proteins_tsne_dbscan.ipynb"

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
            "# MSA Clustering — All Proteins (t-SNE + DBSCAN)",
            "",
            "Batch version of the per-protein MSA-clustering notebooks "
            "(`notebook/msa_clustering/<protein>.ipynb`), restricted to **t-SNE + DBSCAN "
            "only** — t-SNE gave more compact, reliable cluster separation than PCA/UMAP "
            "in practice, so this runs just that half across every protein in one pass "
            "instead of opening 53 separate notebooks. For each protein: one-hot encode "
            "the MSA, t-SNE embed it, auto-select DBSCAN's eps via a k-distance Kneedle "
            "knee, cluster, write per-cluster consensus sequences + `.a3m` subsets + "
            "uniform-control MSAs, and plot the t-SNE landscape (gradient-colored cluster "
            "IDs, no 10-cluster cutoff). A cross-protein summary table is written at the "
            "end.",
            "",
            "Reads every `data/msa/a3m/*.a3m` file (override via the `PROTEINS` parameter "
            "below to run a subset). Outputs go to "
            "`results/msa_clustering/tsne_dbscan/<protein>/` — the same folder used by "
            "the t-SNE+DBSCAN half of the per-protein notebooks, since it's the same "
            "computation.",
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
            "from sklearn.manifold import TSNE",
            "from sklearn.neighbors import NearestNeighbors",
        ),
        code(
            'ROOT = Path("../..")  # notebook/msa_clustering/ is two levels below project root',
            'A3M_DIR = ROOT / "data" / "msa" / "a3m"',
            'RESULTS_DIR = ROOT / "results" / "msa_clustering" / "tsne_dbscan"',
            "",
            "PROTEINS = None          # None = every *.a3m in A3M_DIR; or e.g. [\"NPF6.4_Q9LVE0\", ...] for a subset",
            "",
            "GAP_CUTOFF = 0.25        # drop sequences with > this fraction of gap columns",
            "MIN_SAMPLES = 10         # DBSCAN min_samples (AF-Cluster recommends >= 3, but 3 gave spurious clusters here)",
            "EPS_VAL = None           # set a float to use one fixed eps for every protein instead of per-protein Kneedle",
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
            "def plot_landscape(x, y, df, query_row, label_col, title, outfile, cmap=LANDSCAPE_CMAP):",
            "    fig = plt.figure(figsize=(5, 5))",
            "",
            "    unclustered = df.loc[df[label_col] == -1]",
            '    plt.scatter(unclustered[x], unclustered[y], color="lightgray", marker="x", label="unclustered")',
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
            "One-hot encode → t-SNE embed → Kneedle-selected DBSCAN → per-cluster "
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
            "    n_before = len(df)",
            "    df = df.loc[df.frac_gaps < GAP_CUTOFF].reset_index(drop=True)",
            "    n_after = len(df)",
            "",
            "    ohe_with_query = encode_seqs(df.sequence.tolist() + query.sequence.tolist(), max_len=L)",
            "    tsne_embedding = TSNE(random_state=RANDOM_STATE).fit_transform(ohe_with_query)",
            '    df["TSNE 1"], df["TSNE 2"] = tsne_embedding[:-1, 0], tsne_embedding[:-1, 1]',
            '    query["TSNE 1"], query["TSNE 2"] = tsne_embedding[-1:, 0], tsne_embedding[-1:, 1]',
            "",
            "    labels, eps = select_eps_and_cluster(",
            '        df[["TSNE 1", "TSNE 2"]].values, MIN_SAMPLES, EPS_VAL, "tsne", out_dir, protein',
            "    )",
            '    df["cluster_label_tsne"] = labels',
            "",
            "    clusters = sorted(c for c in df.cluster_label_tsne.unique() if c >= 0)",
            "    n_unclustered = int((df.cluster_label_tsne == -1).sum())",
            '    id_unclustered = avg_identity(df.loc[df.cluster_label_tsne == -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            '    id_clustered = avg_identity(df.loc[df.cluster_label_tsne != -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            "",
            "    records = []",
            "    for clust in clusters:",
            "        tmp = df.loc[df.cluster_label_tsne == clust]",
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
            '                    out_dir / f"{protein}_tsne_{clust:03d}.a3m")',
            "",
            "    metadata_df = pd.DataFrame.from_records(",
            "        records,",
            '        columns=["cluster_ind", "consensusSeq", "avg_lev_dist", "avg_dist_to_query", "size"],',
            "    )",
            "    if len(metadata_df):",
            '        metadata_df = metadata_df.sort_values("size", ascending=False)',
            '    metadata_df.to_csv(out_dir / f"{protein}_tsne_cluster_metadata.tsv", index=False, sep="\\t")',
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
            '    df.to_csv(out_dir / f"{protein}_clustering_assignments.tsv", index=False, sep="\\t")',
            "",
            '    plot_landscape("TSNE 1", "TSNE 2", df, query.iloc[0], "cluster_label_tsne",',
            '                   f"{protein} — t-SNE", out_dir / f"{protein}_TSNE.pdf")',
            "",
            "    return {",
            '        "protein": protein,',
            '        "n_seqs": n_after,',
            '        "L": L,',
            '        "eps": round(eps, 3),',
            '        "n_clusters": len(clusters),',
            '        "n_unclustered": n_unclustered,',
            '        "pct_unclustered": round(n_unclustered / n_after, 4) if n_after else float("nan"),',
            '        "avg_identity_clustered": round(id_clustered, 3) if pd.notna(id_clustered) else id_clustered,',
            '        "avg_identity_unclustered": round(id_unclustered, 3) if pd.notna(id_unclustered) else id_unclustered,',
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
            '        print(f"  eps={summary[\'eps\']:.3f} -> {summary[\'n_clusters\']} clusters, "',
            '              f"{summary[\'n_unclustered\']}/{summary[\'n_seqs\']} unclustered "',
            '              f"({summary[\'pct_unclustered\']:.1%})")',
            "    except Exception as e:",
            '        print(f"  FAILED: {e}")',
            '        summary = {"protein": protein, "error": str(e)}',
            "    summaries.append(summary)",
        ),
        md("## 4. Cross-protein summary"),
        code(
            "summary_df = pd.DataFrame.from_records(summaries)",
            'summary_outfile = RESULTS_DIR / "all_proteins_tsne_dbscan_summary.tsv"',
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
