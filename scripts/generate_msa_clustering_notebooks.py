#!/usr/bin/env python3
"""
scripts/generate_msa_clustering_notebooks.py
=============================================
Stamps out one MSA-clustering notebook per NPF protein under
`notebook/msa_clustering/<protein>.ipynb`.

Each notebook is a self-contained, parameterized (papermill-tag-compatible)
adaptation of the AF-Cluster DBSCAN sequence-clustering workflow
(H Wayment-Steele, 2022): one-hot encode the MSA, project it with PCA / t-SNE /
UMAP, then run DBSCAN (eps auto-selected via a k-distance Kneedle knee) on the
t-SNE and UMAP embeddings — clustering directly on the sparse, very
high-dimensional one-hot space was unreliable, whereas both 2D projections
separate the data much more cleanly. Per-cluster + uniform-control .a3m
subsets and consensus sequences are written out for each embedding's
clustering. Landscape plots color clusters with a continuous gradient
colormap so any number of clusters remains readable (no 10-color cutoff).

Usage:
    python scripts/generate_msa_clustering_notebooks.py
    python scripts/generate_msa_clustering_notebooks.py --protein NPF6.4_Q9LVE0
"""

import argparse
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A3M_DIR = ROOT / "data" / "msa" / "a3m"
NOTEBOOK_DIR = ROOT / "notebook" / "msa_clustering"

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
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags} if tags else {},
        "outputs": [],
        "source": "\n".join(lines),
    }
    return cell


def build_notebook(protein: str) -> dict:
    cells = [
        md(
            f"# MSA Clustering — {protein}",
            "",
            "Sequence-identity clustering of the ColabFold/MMseqs2 MSA for this protein, "
            "adapting the DBSCAN-based approach from Wayment-Steele et al. 2022 "
            "(\"AF-Cluster\"). Sequences are one-hot encoded, then projected with PCA, "
            "t-SNE, and UMAP; DBSCAN is run **on the t-SNE and UMAP embeddings** (not the "
            "raw one-hot features — clustering directly in that sparse, very "
            "high-dimensional space gave unreliable, hard-to-interpret clusters, whereas "
            "both 2D projections separate the MSA much more cleanly). Each cluster's "
            "consensus sequence + per-cluster MSA subset is written out (e.g. as "
            "alternative structure-prediction MSA inputs to sample different "
            "conformational states), for both the t-SNE-based and UMAP-based clustering. "
            "Uniformly-sampled control MSAs are written alongside as a baseline.",
            "",
            f"Reads `data/msa/a3m/{protein}.a3m`. The first record in the file is treated "
            "as the query sequence.",
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
            "from sklearn.decomposition import PCA",
            "from sklearn.manifold import TSNE",
            "from sklearn.neighbors import NearestNeighbors",
            "import umap",
        ),
        code(
            'ROOT = Path("../..")  # notebook/msa_clustering/ is two levels below project root',
            "",
            f'PROTEIN = "{protein}"',
            'A3M_PATH = ROOT / "data" / "msa" / "a3m" / f"{PROTEIN}.a3m"',
            'OUT_DIR = ROOT / "results" / "msa_clustering" / PROTEIN',
            "",
            "GAP_CUTOFF = 0.25         # drop sequences with > this fraction of gap columns",
            "MIN_SAMPLES = 3           # DBSCAN min_samples (AF-Cluster recommends >= 3)",
            "EPS_VAL_TSNE = None       # set a float to skip Kneedle auto-selection for the t-SNE clustering",
            "EPS_VAL_UMAP = None       # set a float to skip Kneedle auto-selection for the UMAP clustering",
            "N_CONTROLS = 10           # number of uniformly-sampled control MSAs per size",
            "RESAMPLE = False          # bootstrap-resample the MSA (with replacement) before clustering",
            "RANDOM_STATE = 0",
            "RUN_PCA = True            # PCA is shown for reference only (not used for clustering)",
            "RUN_TSNE = True           # required for the t-SNE-based clustering below",
            "RUN_UMAP = True           # required for the UMAP-based clustering below",
            'UMAP_METRIC = "jaccard"  # treats each one-hot row as a set of (position, residue) indicators —',
            "                         # matches/mismatches only, and unlike Euclidean it's bounded in [0, 1]",
            "                         # regardless of alignment length, so it needs no per-protein recalibration",
            "UMAP_N_NEIGHBORS = 15",
            "UMAP_MIN_DIST = 0.1",
            'PCA_COLOR_BASIS = "umap"  # which clustering (\"tsne\" or \"umap\") colors the PCA scatter, for comparison',
            'LANDSCAPE_CMAP = "turbo"  # continuous colormap for cluster IDs — no cutoff on number of clusters shown',
            "",
            "OUT_DIR.mkdir(parents=True, exist_ok=True)",
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
            '        kneedle = KneeLocator(np.arange(len(k_dist)), k_dist, curve="convex", direction="increasing")',
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
            "        plt.show()",
            "    else:",
            "        eps = eps_val",
            "",
            "    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_",
            "    n_clust = len(set(c for c in labels if c >= 0))",
            "    n_unclustered = int((labels == -1).sum())",
            '    print(f"[{label}] eps={eps:.3f} -> {n_clust} clusters, {n_unclustered}/{len(labels)} "',
            '          f"unclustered ({n_unclustered/len(labels):.1%})")',
            "    return labels, eps",
        ),
        md("## 2. Load MSA, strip insertions, filter by gap fraction"),
        code(
            "raw_ids, raw_seqs = load_fasta(A3M_PATH)",
            "seqs = strip_insertions(raw_seqs)",
            "",
            'df_all = pd.DataFrame({"SequenceName": raw_ids, "sequence": seqs})',
            "",
            "query = df_all.iloc[:1].copy()",
            "df = df_all.iloc[1:].copy()",
            "",
            "if RESAMPLE:",
            "    df = df.sample(frac=1, random_state=RANDOM_STATE)",
            "",
            "L = len(df.sequence.iloc[0])",
            'print(f"{PROTEIN}: {len(df)} MSA sequences (excluding query), alignment length L={L}")',
            "",
            'df["frac_gaps"] = df.sequence.str.count("-") / L',
            "",
            "n_before = len(df)",
            "df = df.loc[df.frac_gaps < GAP_CUTOFF].reset_index(drop=True)",
            "n_after = len(df)",
            'print(f"Removed {n_before - n_after} seqs with >{int(GAP_CUTOFF*100)}% gaps, {n_after} remaining")',
        ),
        md("## 3. One-hot encode the filtered MSA"),
        code(
            "ohe = encode_seqs(df.sequence.tolist(), max_len=L)",
            'print(f"One-hot encoded shape: {ohe.shape}")',
        ),
        md(
            "## 4. Sequence-landscape embeddings (PCA / t-SNE / UMAP)",
            "",
            "Compute all three projections up front so DBSCAN can be run directly on the "
            "t-SNE and UMAP coordinates below. UMAP uses `UMAP_METRIC = \"jaccard\"`: it "
            "treats each one-hot row as a set of (position, residue) indicators, so "
            "distance only reflects matches/mismatches and, unlike raw Euclidean distance "
            "on the same vectors, is bounded in [0, 1] regardless of alignment length.",
        ),
        code(
            "if RUN_PCA:",
            "    n_pca_components = min(10, ohe.shape[0], ohe.shape[1])",
            "    pca = PCA(n_components=n_pca_components, random_state=RANDOM_STATE)",
            "    pca_embedding = pca.fit_transform(ohe)",
            "    pca_query_embedding = pca.transform(encode_seqs(query.sequence.tolist(), max_len=L))",
            "",
            "    var_explained = pca.explained_variance_ratio_ * 100",
            '    print(f"PCA variance explained — PC1: {var_explained[0]:.1f}%, PC2: {var_explained[1]:.1f}%, "',
            '          f"cumulative (first {n_pca_components} PCs): {var_explained.sum():.1f}%")',
            "",
            "    fig, ax = plt.subplots(figsize=(4, 3))",
            "    ax.bar(np.arange(1, n_pca_components + 1), var_explained)",
            '    ax.set_xlabel("Principal component")',
            '    ax.set_ylabel("Variance explained (%)")',
            '    ax.set_title(f"{PROTEIN} — PCA scree plot")',
            "    plt.tight_layout()",
            '    plt.savefig(OUT_DIR / f"{PROTEIN}_PCA_scree.pdf", bbox_inches="tight")',
            "    plt.show()",
            "",
            '    df["PC 1"], df["PC 2"] = pca_embedding[:, 0], pca_embedding[:, 1]',
            '    query["PC 1"], query["PC 2"] = pca_query_embedding[:, 0], pca_query_embedding[:, 1]',
        ),
        code(
            "if RUN_TSNE:",
            "    ohe_with_query = encode_seqs(df.sequence.tolist() + query.sequence.tolist(), max_len=L)",
            "    tsne_embedding = TSNE(random_state=RANDOM_STATE).fit_transform(ohe_with_query)",
            "",
            '    df["TSNE 1"], df["TSNE 2"] = tsne_embedding[:-1, 0], tsne_embedding[:-1, 1]',
            '    query["TSNE 1"], query["TSNE 2"] = tsne_embedding[-1:, 0], tsne_embedding[-1:, 1]',
        ),
        code(
            "if RUN_UMAP:",
            "    umap_reducer = umap.UMAP(",
            "        metric=UMAP_METRIC,",
            "        n_neighbors=UMAP_N_NEIGHBORS,",
            "        min_dist=UMAP_MIN_DIST,",
            "        random_state=RANDOM_STATE,",
            "    )",
            "    umap_embedding = umap_reducer.fit_transform(ohe)",
            "    umap_query_embedding = umap_reducer.transform(encode_seqs(query.sequence.tolist(), max_len=L))",
            "",
            '    df["UMAP 1"], df["UMAP 2"] = umap_embedding[:, 0], umap_embedding[:, 1]',
            '    query["UMAP 1"], query["UMAP 2"] = umap_query_embedding[:, 0], umap_query_embedding[:, 1]',
        ),
        md(
            "## 5. Cluster the t-SNE embedding",
            "",
            "DBSCAN eps is auto-selected from the k-distance graph of the 2D t-SNE "
            "coordinates via a Kneedle knee (Ester et al. 1996; Satopaa et al. 2011). "
            "Override with `EPS_VAL_TSNE` if the knee plot below looks off.",
        ),
        code(
            "tsne_labels, tsne_eps = select_eps_and_cluster(",
            '    df[["TSNE 1", "TSNE 2"]].values, MIN_SAMPLES, EPS_VAL_TSNE, "tsne", OUT_DIR, PROTEIN',
            ")",
            'df["cluster_label_tsne"] = tsne_labels',
            "",
            'id_unclustered = avg_identity(df.loc[df.cluster_label_tsne == -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            'id_clustered = avg_identity(df.loc[df.cluster_label_tsne != -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            'print(f"Avg identity to query — unclustered: {id_unclustered:.2f}, clustered: {id_clustered:.2f}")',
        ),
        md(
            "## 6. Cluster the UMAP embedding",
            "",
            "Same Kneedle knee-selection approach as above, applied to the 2D UMAP "
            "coordinates. Override with `EPS_VAL_UMAP` if needed.",
        ),
        code(
            "umap_labels, umap_eps = select_eps_and_cluster(",
            '    df[["UMAP 1", "UMAP 2"]].values, MIN_SAMPLES, EPS_VAL_UMAP, "umap", OUT_DIR, PROTEIN',
            ")",
            'df["cluster_label_umap"] = umap_labels',
            "",
            'id_unclustered = avg_identity(df.loc[df.cluster_label_umap == -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            'id_clustered = avg_identity(df.loc[df.cluster_label_umap != -1, "sequence"].tolist(), query.sequence.iloc[0], L)',
            'print(f"Avg identity to query — unclustered: {id_unclustered:.2f}, clustered: {id_clustered:.2f}")',
        ),
        md(
            "## 7. Per-cluster consensus sequences & metadata",
            "",
            "Repeated for both the t-SNE-based and UMAP-based clustering — per-cluster "
            "`.a3m` files and metadata are written separately for each, tagged by basis "
            "(`_tsne_` / `_umap_`) in the filename.",
        ),
        code(
            "cluster_metadata = {}",
            'for basis in ["tsne", "umap"]:',
            '    label_col = f"cluster_label_{basis}"',
            "    clusters = sorted(c for c in df[label_col].unique() if c >= 0)",
            "",
            "    records = []",
            "    for clust in clusters:",
            "        tmp = df.loc[df[label_col] == clust]",
            "        cs = consensus_sequence(tmp.sequence.tolist())",
            "",
            "        avg_dist_to_cs = avg_identity(tmp.sequence.tolist(), cs, L)",
            "        avg_dist_to_query = avg_identity(tmp.sequence.tolist(), query.sequence.iloc[0], L)",
            "",
            "        records.append({",
            '            "cluster_ind": clust,',
            '            "consensusSeq": cs,',
            '            "avg_lev_dist": round(avg_dist_to_cs, 3),',
            '            "avg_dist_to_query": round(avg_dist_to_query, 3),',
            '            "size": len(tmp),',
            "        })",
            "",
            "        cluster_with_query = pd.concat([query, tmp], axis=0)",
            "        write_fasta(cluster_with_query.SequenceName.tolist(), cluster_with_query.sequence.tolist(),",
            '                    OUT_DIR / f"{PROTEIN}_{basis}_{clust:03d}.a3m")',
            "",
            "    metadata_df = pd.DataFrame.from_records(",
            "        records,",
            '        columns=["cluster_ind", "consensusSeq", "avg_lev_dist", "avg_dist_to_query", "size"],',
            "    )",
            "    if len(metadata_df):",
            '        metadata_df = metadata_df.sort_values("size", ascending=False)',
            "    cluster_metadata[basis] = metadata_df",
            "",
            'cluster_metadata["umap"]',
        ),
        md("## 8. Uniform-sample control MSAs"),
        code(
            'print(f"Writing {N_CONTROLS} size-10 uniformly sampled control MSAs")',
            "for i in range(N_CONTROLS):",
            "    tmp = df.sample(n=10, random_state=RANDOM_STATE + i)",
            "    tmp = pd.concat([query, tmp], axis=0)",
            '    write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), OUT_DIR / f"{PROTEIN}_U10-{i:03d}.a3m")',
            "",
            "if len(df) > 100:",
            '    print(f"Writing {N_CONTROLS} size-100 uniformly sampled control MSAs")',
            "    for i in range(N_CONTROLS):",
            "        tmp = df.sample(n=100, random_state=RANDOM_STATE + i)",
            "        tmp = pd.concat([query, tmp], axis=0)",
            '        write_fasta(tmp.SequenceName.tolist(), tmp.sequence.tolist(), OUT_DIR / f"{PROTEIN}_U100-{i:03d}.a3m")',
        ),
        md(
            "## 9. Sequence-landscape plots (PCA / t-SNE / UMAP)",
            "",
            "Each panel is colored by the clustering that was actually run on it: the "
            "t-SNE panel shows `cluster_label_tsne`, the UMAP panel shows "
            "`cluster_label_umap`. The PCA panel (linear, not itself used for clustering) "
            "is colored by `PCA_COLOR_BASIS` purely as a cross-check. Cluster IDs are "
            "mapped through a continuous colormap (`LANDSCAPE_CMAP`) rather than a "
            "10-color palette, so the coloring stays legible regardless of how many "
            "clusters DBSCAN finds.",
        ),
        code(
            "def plot_landscape(x, y, df, query_row, label_col, title, outfile, xlabel=None, ylabel=None,",
            "                    cmap=None):",
            "    cmap = cmap or LANDSCAPE_CMAP",
            "    plt.figure(figsize=(5, 5))",
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
            "    plt.xlabel(xlabel or x)",
            "    plt.ylabel(ylabel or y)",
            "    plt.title(title)",
            "    plt.tight_layout()",
            '    plt.savefig(outfile, bbox_inches="tight")',
            "    plt.show()",
        ),
        code(
            "if RUN_PCA:",
            '    plot_landscape(',
            '        "PC 1", "PC 2", df, query.iloc[0], f"cluster_label_{PCA_COLOR_BASIS}",',
            '        f"{PROTEIN} — PCA (colored by {PCA_COLOR_BASIS} clusters)", OUT_DIR / f"{PROTEIN}_PCA.pdf",',
            '        xlabel=f"PC 1 ({var_explained[0]:.1f}% var.)",',
            '        ylabel=f"PC 2 ({var_explained[1]:.1f}% var.)",',
            "    )",
        ),
        code(
            "if RUN_TSNE:",
            '    plot_landscape("TSNE 1", "TSNE 2", df, query.iloc[0], "cluster_label_tsne",',
            '                   f"{PROTEIN} — t-SNE", OUT_DIR / f"{PROTEIN}_TSNE.pdf")',
        ),
        code(
            "if RUN_UMAP:",
            '    plot_landscape("UMAP 1", "UMAP 2", df, query.iloc[0], "cluster_label_umap",',
            '                   f"{PROTEIN} — UMAP ({UMAP_METRIC})", OUT_DIR / f"{PROTEIN}_UMAP.pdf")',
        ),
        md("## 10. Save clustering assignments & metadata tables"),
        code(
            'assignments_outfile = OUT_DIR / f"{PROTEIN}_clustering_assignments.tsv"',
            'df.to_csv(assignments_outfile, index=False, sep="\\t")',
            'print(f"Wrote clustering assignments to {assignments_outfile}")',
            "",
            'for basis, metadata_df in cluster_metadata.items():',
            '    metadata_outfile = OUT_DIR / f"{PROTEIN}_{basis}_cluster_metadata.tsv"',
            '    metadata_df.to_csv(metadata_outfile, index=False, sep="\\t")',
            '    print(f"Wrote {basis} cluster metadata to {metadata_outfile}")',
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protein", action="store", help="Generate a single protein's notebook (default: all found in data/msa/a3m/).")
    args = p.parse_args()

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    if args.protein:
        proteins = [args.protein]
    else:
        proteins = sorted(f.stem for f in A3M_DIR.glob("*.a3m"))

    for protein in proteins:
        nb = build_notebook(protein)
        outfile = NOTEBOOK_DIR / f"{protein}.ipynb"
        outfile.write_text(json.dumps(nb, indent=1))
        print(f"wrote {outfile}")

    print(f"Generated {len(proteins)} notebook(s) in {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
