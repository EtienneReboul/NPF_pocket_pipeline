#!/usr/bin/env python3
"""
src/vina_score.py
====================
Score ONE complex's current pose with AutoDock Vina
(https://autodock-vina.readthedocs.io/en/latest/docking_python.html).

Deliberately scores the pose AS-IS (`v.score()`) rather than docking/
searching — this mirrors the request to "score the current pose", not
re-dock it. `v.optimize()` (a quick local BFGS minimization, Vina's analogue
of the PyRosetta pipeline's light relax) is also recorded for comparison —
same raw-vs-relaxed pattern as run_complex.py.

Column meanings (Vina scoring function; see `vina.Vina.score`/`.optimize`
docstrings): [total, lig_inter, flex_inter, other_inter, flex_intra,
lig_intra, torsions, lig_intra_best_pose]. Units are kcal/mol-scale (Vina's
own empirical function, calibrated to roughly track binding affinity) —
unlike Rosetta's REU, but still an empirical scoring function, not a
rigorous physical free energy.

Usage:
    python src/vina_score.py --complex-id <id from manifest.csv>
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from vina import Vina

import config
import pose_prep as pp
import vina_prep as vp

SCORE_COLS = ["total", "lig_inter", "flex_inter", "other_inter", "flex_intra", "lig_intra", "torsions", "lig_intra_best"]


def run_one(complex_id: str, pdb_path: Path, protein: str, cls: str,
            naming: dict, template, log_file=None) -> pd.DataFrame:
    def log(msg):
        print(msg, file=log_file or sys.stdout, flush=True)

    t0 = time.time()
    text = pdb_path.read_text()
    out_prefix = config.VINA_PDBQT_DIR / complex_id

    ligand_pdbqt_str, coords, anomaly = vp.prepare_ligand_pdbqt(text, template, naming)
    ligand_pdbqt_path = Path(f"{out_prefix}_ligand.pdbqt")
    ligand_pdbqt_path.write_text(ligand_pdbqt_str)
    log(f"[{complex_id}] ligand PDBQT written, raw anomaly={anomaly}")

    receptor_pdbqt_path, receptor_log = vp.prepare_receptor_pdbqt(text, Path(f"{out_prefix}_receptor"))
    receptor_warning = "Files written" not in receptor_log
    if receptor_warning:
        log(f"[{complex_id}] mk_prepare_receptor.py output (flagged, no success message):\n{receptor_log}")

    center, box_size = vp.ligand_box(coords)

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(receptor_pdbqt_path))
    v.set_ligand_from_file(str(ligand_pdbqt_path))
    v.compute_vina_maps(center=center, box_size=box_size)

    raw = v.score()
    optimized = v.optimize()
    log(f"[{complex_id}] total {raw[0]:.3f} -> {optimized[0]:.3f} kcal/mol "
        f"(lig_inter {raw[1]:.3f} -> {optimized[1]:.3f})")

    row = {"complex_id": complex_id, "protein": protein, "class": cls}
    for name, val in zip(SCORE_COLS, raw):
        row[name] = val
    for name, val in zip(SCORE_COLS, optimized):
        row[f"{name}_optimized"] = val
    row.update({
        "n_hetatm_raw": anomaly["n_hetatm"],
        "n_heavy_heavy_bonds_raw": anomaly["n_heavy_heavy_bonds"],
        "hetatm_count_anomaly": anomaly["hetatm_count_anomaly"],
        "bond_count_anomaly": anomaly["bond_count_anomaly"],
        "receptor_prep_warning": receptor_warning,
    })
    log(f"[{complex_id}] done in {time.time() - t0:.1f}s")
    return pd.DataFrame([row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex-id", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(config.MANIFEST_CSV)
    rows = manifest[manifest["complex_id"] == args.complex_id]
    if rows.empty:
        sys.exit(f"complex_id {args.complex_id!r} not found in {config.MANIFEST_CSV}")
    row = rows.iloc[0]

    naming = pp.load_atom_naming()
    template = pp.load_template(naming)

    df = run_one(row["complex_id"], config.PIPELINE_ROOT / row["pdb_path"], row["protein"], row["class"],
                 naming, template)
    out_path = config.VINA_PER_COMPLEX_DIR / f"{row['complex_id']}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
