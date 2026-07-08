#!/usr/bin/env python3
"""
scripts/generate_msa_clustering_blosum62_tsne_hdbscan_all_proteins_notebook.py
================================================================================
Builds a single notebook, `notebook/msa_clustering/all_proteins_blosum62_tsne_hdbscan.ipynb`,
that batch-processes every NPF protein under `data/msa/a3m/` with the same
BLOSUM62 + DBCV-tuned HDBSCAN combination as
`generate_msa_clustering_blosum62_umap_hdbscan_all_proteins_notebook.py`, but
swapping the embedding step for **t-SNE** instead of UMAP: sequences are
encoded with a BLOSUM62-like numeric embedding (adapted from an external
`cluster-msa.py` reference script's `encode_seqs_bl62` / `BL62NP` — each
residue mapped to a 16-dim substitution-based vector) instead of one-hot
encoding, embedded with sklearn's `TSNE` to `TSNE_N_COMPONENTS` dimensions
(2 by default — unlike UMAP, sklearn's default Barnes-Hut TSNE only supports
up to 3 components, so HDBSCAN clusters directly in the same 2D space used
for plotting, matching the convention of the other t-SNE notebooks in this
repo), and clustered with HDBSCAN whose hyperparameters (`min_samples`,
`min_cluster_size`, `cluster_selection_method`, `metric`) are tuned per
protein via a random search scored by DBCV (Density-Based Clustering
Validation, Moulavi et al. 2014) — unlike Silhouette Score, DBCV accounts for
noise and measures density rather than distance, which actually matches a
density-based algorithm's assumptions.

Note on t-SNE vs. UMAP: `sklearn.manifold.TSNE` has no `.transform()` method
for embedding new points into an existing map, unlike `umap.UMAP`. So the
query sequence is concatenated onto the rest of the MSA before the single
`fit_transform` call (same approach as `generate_msa_clustering_tsne_*`
notebooks), then split back out afterwards.

Note on the hyperparameter search: the blog post this convention is drawn
from drives the search with sklearn's `RandomizedSearchCV`, which by default
cross-validates by splitting rows into folds. That doesn't make sense here —
folding a single MSA's embedding fragments its density structure before
HDBSCAN ever sees the whole picture. Instead, each sampled hyperparameter
combination is fit once on the *entire* per-protein embedding and scored with
`hdbscan.validity.validity_index` directly; the combination with the highest
DBCV is refit and used for the final clustering. Combinations that collapse
to fewer than 2 clusters, or that error out inside DBCV, are scored as the
worst possible so they're never selected.

Per-cluster consensus + `.a3m` subsets and uniform-control MSAs are written
out, plus a gradient-colored t-SNE landscape plot, a DBCV-vs-hyperparameter-
combination diagnostic plot, and a cross-protein summary table.

Usage:
    python scripts/generate_msa_clustering_blosum62_tsne_hdbscan_all_proteins_notebook.py

Runtime note: each protein fits HDBSCAN + computes DBCV up to
`N_HYPERPARAM_SEARCH_ITER` times (default 60); DBCV's internal pairwise
computation is the dominant cost and grows faster than linearly with MSA
size, so the full 53-protein batch will take noticeably longer than the
DBSCAN-only notebooks.
"""

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "notebook" / "msa_clustering" / "all_proteins_blosum62_tsne_hdbscan.ipynb"

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


BL62NP_LITERAL = '''BL62NP = {
    "A": [
        -0.31230882, -0.53572156, -0.01949946, -0.12211268, -0.70947917,
        -0.42211092, 0.02783931, 0.02637933, -0.41760305, 0.21809875,
        0.53532768, 0.04833016, 0.07877711, 0.50464914, -0.26972087, -0.52416842,
    ],
    "R": [
        0.29672002, 0.29005364, 0.18176298, -0.05103382, -0.34686519,
        0.58024228, -0.49282931, 0.62304281, -0.09575202, 0.30115555,
        0.09913529, 0.1577466, -0.94391939, -0.10505925, 0.05482389, 0.38409897,
    ],
    "N": [
        -0.42212537, 0.12225749, 0.16279646, 0.60099009, 0.19734216,
        0.42819919, -0.33562418, 0.17036334, 0.4234109, 0.46681561,
        -0.50347222, -0.37936876, 0.1494825, 0.32176759, 0.28584684, 0.68469861,
    ],
    "D": [
        0.18599294, -0.44017825, -0.4476952, 0.34340976, 0.44603553,
        0.40974629, -0.60045935, -0.09056728, 0.22147919, -0.33029418,
        0.55635594, -0.54149972, 0.05459062, 0.57334159, -0.06227118, 0.65299872,
    ],
    "C": [
        -0.19010428, 0.64418792, -0.85286762, 0.21380295, 0.37639516,
        -0.67753593, 0.38751609, 0.55746524, 0.01443766, 0.1776535,
        0.62853954, -0.15048523, 0.55100206, -0.21426656, 0.3644061, -0.0018255,
    ],
    "Q": [
        0.7350723, 0.10111267, 0.55640019, -0.18226966, 0.51658102,
        -0.19321508, -0.46599027, -0.02989911, 0.4036196, -0.11978213,
        -0.29837524, -0.30232765, -0.36738065, -0.1379793, 0.04362871, 0.33553714,
    ],
    "E": [
        0.41134047, 0.13512443, 0.62492322, -0.10120261, -0.03093491,
        0.23751917, -0.68338694, 0.05124762, 0.41533821, 0.46669353,
        0.31467277, -0.02427587, 0.15361135, 0.70595112, -0.27952632, 0.32408931,
    ],
    "G": [
        -0.33041265, -0.43860065, -0.5509376, -0.04380843, -0.35160935,
        0.25134855, 0.53409314, 0.54850824, 0.59490287, 0.32669345,
        -0.45355268, -0.56317041, -0.55416297, 0.18117841, -0.71600849, -0.08989825,
    ],
    "H": [
        -0.40366849, 0.10978974, 0.0280101, -0.46667987, -0.45607028,
        0.54114052, -0.77552923, -0.10720425, 0.55252091, -0.34397153,
        -0.59813694, 0.15567728, 0.03071009, -0.02176143, 0.34442719, 0.14681541,
    ],
    "I": [
        0.19280422, 0.35777863, 0.06139255, 0.20081699, -0.30546596,
        -0.56901549, -0.15290953, -0.31181573, -0.74523217, 0.22296016,
        -0.39143832, -0.16474685, 0.58064427, -0.77386654, 0.19713107, -0.49477418,
    ],
    "L": [
        -0.16133903, 0.22112761, -0.53162136, 0.34764073, -0.08522381,
        -0.2510216, 0.04699411, -0.25702389, -0.8739765, -0.24171728,
        -0.24370533, 0.42193635, 0.41056913, -0.60378211, -0.65756832, 0.0845203,
    ],
    "K": [
        -0.34792144, 0.18450939, 0.77038332, 0.63868511, -0.06221681,
        0.11930421, 0.04895523, -0.22463059, -0.03268844, -0.58941354,
        0.11640045, 0.32384901, -0.42952779, 0.58119471, 0.07288662, 0.26669673,
    ],
    "M": [
        0.01834555, -0.16367754, 0.34900298, 0.45087949, 0.47073855,
        -0.37377404, 0.0606911, 0.2455703, -0.55182937, -0.20261009,
        0.28325423, -0.04741146, 0.30565238, -0.62090653, 0.17528413, -0.60434975,
    ],
    "F": [
        -0.55464981, 0.50918784, -0.21371646, -0.63996967, -0.37656862,
        0.27852662, 0.3287838, -0.56800869, 0.23260763, -0.20653106,
        0.63261439, -0.22666691, 0.00726302, -0.60125196, 0.07139961, -0.35086639,
    ],
    "P": [
        0.94039731, -0.25999326, 0.43922549, -0.485738, -0.20492235,
        -0.26005626, 0.68776626, 0.57826888, -0.05973995, -0.1193658,
        -0.12102433, -0.22091354, 0.43427913, 0.71447886, 0.32745991, 0.03466398,
    ],
    "S": [
        -0.13194625, -0.12262688, 0.18029209, 0.16555524, 0.39594125,
        -0.58110665, 0.16161717, 0.0839783, 0.0911945, 0.34546976,
        -0.29415349, 0.29891936, -0.60834721, 0.5943593, -0.29473819, 0.4864154,
    ],
    "T": [
        0.40850093, -0.4638894, -0.39732987, -0.01972861, 0.51189582,
        0.10176704, 0.37528519, -0.41479418, -0.1932531, 0.54732221,
        -0.11876511, 0.32843973, -0.259283, 0.59500132, 0.35168375, -0.21733727,
    ],
    "W": [
        -0.50627723, -0.1973602, -0.02339884, -0.66846048, 0.62696606,
        0.60049717, 0.69143364, -0.48053591, 0.17812208, -0.58481821,
        -0.23551415, -0.06229112, 0.20993116, -0.72485884, 0.34375662, -0.23539168,
    ],
    "Y": [
        -0.51388312, -0.2788953, 0.00859533, -0.5247195, -0.18021544,
        0.28372911, 0.10791359, 0.13033494, 0.34294013, -0.70310089,
        -0.13245433, 0.48661081, 0.08451644, -0.69990992, 0.0408274, -0.47204888,
    ],
    "V": [
        0.68546275, 0.22581365, -0.32571833, 0.34394298, -0.43232367,
        -0.5041842, 0.04784017, -0.53067936, -0.50049908, 0.36874221,
        0.22429186, 0.4616482, 0.11159174, -0.26827959, -0.39372848, -0.40987423,
    ],
}
BL62NP["-"] = np.mean(list(BL62NP.values()), axis=0).tolist()'''


def build_notebook() -> dict:
    cells = [
        md(
            "# MSA Clustering — All Proteins (BLOSUM62 + t-SNE + DBCV-tuned HDBSCAN)",
            "",
            "Same BLOSUM62 + DBCV-tuned HDBSCAN combination as "
            "`all_proteins_blosum62_umap_hdbscan.ipynb`, but with **t-SNE** in place "
            "of UMAP for the embedding step: sequences are encoded with a "
            "**BLOSUM62-like numeric embedding** (adapted from an external "
            "`cluster-msa.py` reference script's `encode_seqs_bl62` / `BL62NP` — each "
            "residue mapped to a 16-dim substitution-based vector) instead of "
            "one-hot encoding, embedded with sklearn's **TSNE** to "
            "`TSNE_N_COMPONENTS` dimensions (2 by default — unlike UMAP, sklearn's "
            "default Barnes-Hut TSNE only supports up to 3 components, so HDBSCAN "
            "clusters directly in the same 2D space used for plotting, matching the "
            "convention of the other t-SNE notebooks in this repo), and clustered "
            "with **HDBSCAN** whose `min_samples` / `min_cluster_size` / "
            "`cluster_selection_method` / `metric` are tuned per protein via a "
            "random hyperparameter search scored by **DBCV** (Density-Based "
            "Clustering Validation, Moulavi et al. 2014) — unlike Silhouette Score, "
            "DBCV accounts for noise and measures density rather than distance, "
            "matching a density-based algorithm's assumptions (see \"On the "
            "Validation of HDBSCAN\").",
            "",
            "**t-SNE vs. UMAP:** `sklearn.manifold.TSNE` has no `.transform()` "
            "method for embedding new points into an existing map, unlike "
            "`umap.UMAP`. So the query sequence is concatenated onto the rest of "
            "the MSA before a single `fit_transform` call, then split back out "
            "afterwards — same approach as the other t-SNE notebooks in this repo.",
            "",
            "Each sampled hyperparameter combination is fit once on the **entire** "
            "per-protein embedding (not cross-validated folds — folding a single MSA's "
            "embedding would fragment its density structure before HDBSCAN ever sees "
            "the whole picture) and scored with `hdbscan.validity.validity_index` "
            "directly; the combination with the highest DBCV is refit and used for the "
            "final clustering. Combinations collapsing to fewer than 2 clusters, or "
            "erroring inside DBCV, score as the worst possible so they're never "
            "selected.",
            "",
            "Per-cluster consensus sequences + `.a3m` subsets + uniform-control MSAs "
            "are written out, along with a gradient-colored t-SNE landscape plot, a "
            "DBCV-vs-hyperparameter-combo diagnostic plot, and a cross-protein "
            "summary table.",
            "",
            "Reads every `data/msa/a3m/*.a3m` file (override via the `PROTEINS` "
            "parameter below to run a subset). Outputs go to "
            "`results/msa_clustering/blosum62_tsne_hdbscan/<protein>/`, its own "
            "folder so they don't collide with the other MSA-clustering notebooks' "
            "outputs.",
            "",
            "**Runtime note:** each protein fits HDBSCAN + computes DBCV up to "
            "`N_HYPERPARAM_SEARCH_ITER` times (default 60); DBCV's internal pairwise "
            "computation is the dominant cost and grows faster than linearly with MSA "
            "size, so the full batch takes noticeably longer than the DBSCAN-only "
            "notebooks.",
        ),
        code(
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "from pathlib import Path",
            "from Bio import SeqIO",
            "from polyleven import levenshtein",
            "from sklearn.cluster import HDBSCAN",
            "from sklearn.manifold import TSNE",
            "from hdbscan.validity import validity_index",
        ),
        code(
            'ROOT = Path("../..")  # notebook/msa_clustering/ is two levels below project root',
            'A3M_DIR = ROOT / "data" / "msa" / "a3m"',
            'RESULTS_DIR = ROOT / "results" / "msa_clustering" / "blosum62_tsne_hdbscan"',
            "",
            "PROTEINS = None          # None = every *.a3m in A3M_DIR; or e.g. [\"NPF6.4_Q9LVE0\", ...] for a subset",
            "",
            "GAP_CUTOFF = 0.25            # drop sequences with > this fraction of gap columns",
            "TSNE_N_COMPONENTS = 2        # sklearn's default Barnes-Hut TSNE supports at most 3",
            "TSNE_PERPLEXITY = 30.0       # sklearn default",
            "HDBSCAN_MIN_SAMPLES_CANDIDATES = [3, 5, 10, 15, 20, 25, 30, 40, 50]",
            "HDBSCAN_MIN_CLUSTER_SIZE_CANDIDATES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150]",
            'HDBSCAN_CLUSTER_SELECTION_METHODS = ["eom", "leaf"]',
            "HDBSCAN_METRICS = [\"euclidean\", \"cityblock\"]  # not \"manhattan\": same distance, but that",
            "                                              # name errors inside hdbscan.validity.validity_index",
            "N_HYPERPARAM_SEARCH_ITER = 60    # random (min_samples, min_cluster_size, method, metric) combos tried per protein",
            "N_CONTROLS = 10          # number of uniformly-sampled control MSAs per size",
            "RESAMPLE = False         # bootstrap-resample each MSA (with replacement) before clustering",
            "RANDOM_STATE = 0",
            'LANDSCAPE_CMAP = "turbo"  # continuous colormap for cluster IDs — no cutoff on number of clusters shown',
            "",
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)",
            "np.random.seed(RANDOM_STATE)",
            tags=["parameters"],
        ),
        md(
            "## 1. Helper functions",
            "",
            "`BL62NP` is the BLOSUM62-like per-residue embedding table from an "
            "external `cluster-msa.py` reference script, copied verbatim so this "
            "notebook stays self-contained.",
        ),
        code(BL62NP_LITERAL),
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
            "def encode_seqs_bl62(seqs, max_len, bl62=BL62NP):",
            '    """BLOSUM62-like numeric embedding (adapted from an external cluster-msa.py reference',
            "    script's encode_seqs_bl62): each residue is",
            "    replaced by its 16-dim substitution-based vector (gaps get their own averaged vector),",
            '    rows padded to max_len with gaps, flattened to (n_seqs, max_len * 16)."""',
            '    dim = len(bl62["-"])',
            "    arr = np.array([",
            '        [bl62.get(c, bl62["-"]) for c in seq.upper().ljust(max_len, "-")]',
            "        for seq in seqs",
            "    ])",
            "    return arr.reshape(len(seqs), max_len * dim)",
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
            "def select_hdbscan_hyperparams_and_cluster(X, min_samples_candidates, min_cluster_size_candidates,",
            "                                            cluster_selection_methods, metrics, n_iter,",
            "                                            random_state, label, out_dir, protein):",
            '    """Randomly samples n_iter (min_samples, min_cluster_size, cluster_selection_method, metric)',
            "    combinations, fits HDBSCAN for each directly on the full embedding X (not cross-validated —",
            "    density-based structure doesn't survive being split into folds), and scores each with DBCV",
            "    (Moulavi et al. 2014) via hdbscan.validity.validity_index, which — unlike Silhouette Score —",
            "    accounts for noise and uses density rather than distance, matching the assumptions of a",
            "    density-based clustering algorithm. The combination with the highest DBCV is refit and",
            "    returned. Combos yielding fewer than 2 clusters, or that error inside DBCV, score -1.0 so",
            '    they\'re never selected."""',
            "    max_min_cluster_size = max(2, len(X) // 5)",
            "    candidate_min_cluster_sizes = [m for m in min_cluster_size_candidates if 2 <= m <= max_min_cluster_size]",
            "    if not candidate_min_cluster_sizes:",
            "        candidate_min_cluster_sizes = [max_min_cluster_size]",
            "",
            "    grid = [",
            "        (ms, mcs, csm, metric)",
            "        for ms in min_samples_candidates",
            "        for mcs in candidate_min_cluster_sizes",
            "        for csm in cluster_selection_methods",
            "        for metric in metrics",
            "    ]",
            "    rng = np.random.RandomState(random_state)",
            "    n_try = min(n_iter, len(grid))",
            "    combos = [grid[i] for i in rng.choice(len(grid), size=n_try, replace=False)]",
            "",
            "    results = []",
            "    for min_samples, min_cluster_size, cluster_selection_method, metric in combos:",
            "        try:",
            "            labels = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size,",
            "                              cluster_selection_method=cluster_selection_method,",
            "                              metric=metric, copy=False).fit(X).labels_",
            "            n_clust = len(set(c for c in labels if c >= 0))",
            "            dbcv = float(validity_index(X.astype(np.float64), labels, metric=metric)) if n_clust >= 2 else -1.0",
            "        except Exception as e:",
            '            print(f"[{label}] combo ms={min_samples} mcs={min_cluster_size} "',
            '                  f"{cluster_selection_method}/{metric} failed: {e}")',
            "            labels, dbcv = None, -1.0",
            "        results.append({",
            '            "min_samples": min_samples, "min_cluster_size": min_cluster_size,',
            '            "cluster_selection_method": cluster_selection_method, "metric": metric,',
            '            "dbcv": dbcv, "labels": labels,',
            "        })",
            "",
            '    best = max(results, key=lambda r: r["dbcv"])',
            "",
            '    search_df = pd.DataFrame([{k: v for k, v in r.items() if k != "labels"} for r in results])',
            '    search_df = search_df.sort_values("dbcv", ascending=False)',
            '    search_df.to_csv(out_dir / f"{protein}_{label}_hyperparam_search.tsv", index=False, sep="\\t")',
            "",
            "    fig, ax = plt.subplots(figsize=(5, 4))",
            '    order = np.arange(len(search_df))',
            '    ax.bar(order, search_df["dbcv"], color="steelblue")',
            '    ax.axhline(best["dbcv"], color="red", ls="--", label=f"best DBCV={best[\'dbcv\']:.3f}")',
            '    ax.set_xlabel("hyperparameter combination (sorted by DBCV)")',
            '    ax.set_ylabel("DBCV (relative validity)")',
            '    ax.set_title(f"{protein} — {label} hyperparameter search")',
            "    ax.legend()",
            "    plt.tight_layout()",
            '    plt.savefig(out_dir / f"{protein}_{label}_dbcv_search.pdf", bbox_inches="tight")',
            "    plt.close(fig)",
            "",
            '    return best["labels"], best',
            "",
            "",
            "def plot_landscape(x, y, df, query_row, label_col, title, outfile, cmap=LANDSCAPE_CMAP):",
            "    fig = plt.figure(figsize=(5, 5))",
            "",
            "    unclustered = df.loc[df[label_col] == -1]",
            "    if len(unclustered):",
            '        plt.scatter(unclustered[x], unclustered[y], color="lightgray", marker="x", label="noise (HDBSCAN)")',
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
            "BLOSUM62-encode → t-SNE embed (query concatenated in, since `TSNE` has "
            "no out-of-sample `.transform()`) → DBCV-tuned HDBSCAN on the "
            "`TSNE_N_COMPONENTS`-D embedding → per-cluster consensus/`.a3m` + "
            "uniform controls + landscape plot. Returns a one-row summary dict used "
            "to build the cross-protein table in the last section.",
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
            "    bl62_with_query = encode_seqs_bl62(df.sequence.tolist() + query.sequence.tolist(), max_len=L)",
            "    tsne_embedding = TSNE(n_components=TSNE_N_COMPONENTS, perplexity=TSNE_PERPLEXITY,",
            "                          random_state=RANDOM_STATE).fit_transform(bl62_with_query)",
            "    tsne_cols = [f\"TSNE {i + 1}\" for i in range(TSNE_N_COMPONENTS)]",
            "    for i, col in enumerate(tsne_cols):",
            "        df[col] = tsne_embedding[:-1, i]",
            "        query[col] = tsne_embedding[-1:, i]",
            "",
            "    hdbscan_labels, best = select_hdbscan_hyperparams_and_cluster(",
            "        df[tsne_cols].values, HDBSCAN_MIN_SAMPLES_CANDIDATES, HDBSCAN_MIN_CLUSTER_SIZE_CANDIDATES,",
            "        HDBSCAN_CLUSTER_SELECTION_METHODS, HDBSCAN_METRICS, N_HYPERPARAM_SEARCH_ITER,",
            '        RANDOM_STATE, "hdbscan", out_dir, protein,',
            "    )",
            '    df["cluster_label_hdbscan"] = hdbscan_labels',
            "    n_clusters = len(sorted(c for c in df.cluster_label_hdbscan.unique() if c >= 0))",
            '    n_noise = int((df.cluster_label_hdbscan == -1).sum())',
            "",
            '    avg_id_to_query = avg_identity(df.sequence.tolist(), query.sequence.iloc[0], L)',
            '    id_noise = avg_identity(df.loc[df.cluster_label_hdbscan == -1, "sequence"].tolist(),',
            "                            query.sequence.iloc[0], L)",
            '    id_clustered = avg_identity(df.loc[df.cluster_label_hdbscan != -1, "sequence"].tolist(),',
            "                                query.sequence.iloc[0], L)",
            "",
            "    clusters = sorted(c for c in df.cluster_label_hdbscan.unique() if c >= 0)",
            "    records = []",
            "    for clust in clusters:",
            "        tmp = df.loc[df.cluster_label_hdbscan == clust]",
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
            '                    out_dir / f"{protein}_hdbscan_{clust:03d}.a3m")',
            "",
            "    metadata_df = pd.DataFrame.from_records(",
            "        records,",
            '        columns=["cluster_ind", "consensusSeq", "avg_lev_dist", "avg_dist_to_query", "size"],',
            "    )",
            "    if len(metadata_df):",
            '        metadata_df = metadata_df.sort_values("size", ascending=False)',
            '    metadata_df.to_csv(out_dir / f"{protein}_hdbscan_cluster_metadata.tsv", index=False, sep="\\t")',
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
            '    df.to_csv(out_dir / f"{protein}_hdbscan_clustering_assignments.tsv", index=False, sep="\\t")',
            "",
            '    plot_landscape("TSNE 1", "TSNE 2", df, query.iloc[0], "cluster_label_hdbscan",',
            '                   f"{protein} — BLOSUM62+TSNE+HDBSCAN (DBCV={best[\'dbcv\']:.2f})",',
            '                   out_dir / f"{protein}_TSNE_HDBSCAN.pdf")',
            "",
            "    return {",
            '        "protein": protein,',
            '        "n_seqs": n_after,',
            '        "L": L,',
            '        "best_min_samples": best["min_samples"],',
            '        "best_min_cluster_size": best["min_cluster_size"],',
            '        "best_cluster_selection_method": best["cluster_selection_method"],',
            '        "best_metric": best["metric"],',
            '        "best_dbcv": round(best["dbcv"], 4),',
            '        "n_clusters": n_clusters,',
            '        "n_noise": n_noise,',
            '        "pct_noise": round(n_noise / n_after, 4) if n_after else float("nan"),',
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
            '        print(f"  HDBSCAN: ms={summary[\'best_min_samples\']} mcs={summary[\'best_min_cluster_size\']} "',
            '              f"{summary[\'best_cluster_selection_method\']}/{summary[\'best_metric\']} "',
            '              f"-> DBCV={summary[\'best_dbcv\']:.3f}, {summary[\'n_clusters\']} clusters, "',
            '              f"{summary[\'n_noise\']}/{summary[\'n_seqs\']} noise ({summary[\'pct_noise\']:.1%})")',
            "    except Exception as e:",
            '        print(f"  FAILED: {e}")',
            '        summary = {"protein": protein, "error": str(e)}',
            "    summaries.append(summary)",
        ),
        md("## 4. Cross-protein summary"),
        code(
            "summary_df = pd.DataFrame.from_records(summaries)",
            'summary_outfile = RESULTS_DIR / "all_proteins_blosum62_tsne_hdbscan_summary.tsv"',
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
