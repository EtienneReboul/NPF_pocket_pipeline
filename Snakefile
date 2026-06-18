"""
NPF Family Structure Modelling Pipeline — Master Snakefile
==========================================================

Stages
------
  1.  msa            — UniProt download + ColabFold MSA per NPF member
  2.  interproscan   — EMBL-EBI CDD annotation + binding-site residue extraction
  2b. tm_annotation  — EMBL-EBI Phobius + TMHMM TM helix topology annotation
  3.  templates      — RCSB mmCIF download per conformation set
  4.  boltz_input    — generate Boltz-2 YAML per protein × conformation
  5.  boltz_run      — run Boltz-2 per protein × conformation
  6.  minimize       — ChimeraX energy minimization, mmCIF → PDB per diffusion sample
  7.  plip           — PLIP via Docker per sample
  8.  pliparser      — PLIP report → CSV + ChimeraX .cxc visualisation
  9.  aggregate      — merge per-sample CSVs into one CSV per protein × conformation
  10. dssp           — ChimeraX DSSP on raw Boltz-2 CIF (sanity check)
  11. tm_angle       — TM2/TM8 helix axis angle per sample
  12. collect_angles — merge per-sample angles into one CSV per protein × conformation
  13. gmm_analysis   — fit 3- and 6-component GMMs, compare by BIC

Output tree
-----------
  results/
    boltz/{protein}/{conformation}/
      boltz_out/                       ← Boltz-2 mmCIF outputs
    minimized/{protein}/{conformation}/
      {sample_id}/
        model_minimized.pdb
        model_minimized_energy.csv
    plip/{protein}/{conformation}/
      {sample_id}/
        model_minimized_report/
          model_minimized_report.txt
          csv/summary.csv
          interaction.cxc
      summary.csv                      ← ★ aggregated across all samples
    dssp/{protein}/{conformation}/
      {sample_id}/dssp.json            ← per-residue DSSP SS from Boltz-2 CIF
    tm_angles/{protein}/{conformation}/
      {sample_id}/angle.csv            ← TM2/TM8 angle for one sample
      angles.csv                       ← ★ aggregated across all samples
    gmm/
      gmm_report.json                  ← BIC, means, weights, interpretation
      angles_with_assignments.csv      ← all valid angles + GMM labels
      gmm3.png / gmm6.png              ← density plots
      bic_comparison.png               ← bar chart BIC(3) vs BIC(6)
      angle_by_conformation.png        ← strip-plot coloured by GMM-3 component
"""

import json
import re
from pathlib import Path
from collections import defaultdict


configfile: "config.yaml"


# ── Directory shortcuts ────────────────────────────────────────────────────────
DIRS = config["dirs"]
FASTA_DIR = Path(DIRS["fasta"])
MSA_DIR = Path(DIRS["msa"])
INTERPRO_DIR = Path(DIRS["interpro"])
TMPL_DIR = Path(DIRS["templates"])
BOLTZ_IN_DIR = Path(DIRS["boltz_in"])
BOLTZ_OUT = Path(DIRS["boltz_out"])
MIN_DIR = Path(DIRS["minimized"])
PLIP_DIR = Path(DIRS["plip"])
DSSP_DIR = Path(DIRS["dssp"])
TM_ANG_DIR = Path(DIRS["tm_angles"])
GMM_DIR = Path(DIRS["gmm"])
LOG_DIR = Path(DIRS["logs"])

# ── Config shortcuts ───────────────────────────────────────────────────────────
BOLTZ_CFG = config["boltz"]
PLIP_CFG = config["plip"]
VIZ_CFG = config["visualization"]
TM_ANN_CFG = config["tm_annotation"]
TM_ANG_CFG = config["tm_angle"]
GMM_CFG = config["gmm"]
CHIMERAX_BIN = config["chimerax_bin"]
CONFORMATIONS = list(config["templates"]["conformations"].keys())

# ── Lazy wildcard discovery ────────────────────────────────────────────────────
# Proteins are discovered after Stage 1 completes (sentinel file).
# The sentinel lists one protein base-name per line.
# We use a checkpoint so Snakemake re-evaluates the DAG after MSA finishes.

MSA_SENTINEL = MSA_DIR / "msa.done"
IPRO_SENTINEL = INTERPRO_DIR / "interproscan.done"
TM_ANN_SENTINEL = INTERPRO_DIR / "tm_annotation.done"
TMPL_SENTINEL = TMPL_DIR / "templates.done"


def read_proteins(sentinel_path):
    """Read protein names from the MSA sentinel file."""
    p = Path(sentinel_path)
    if p.exists():
        return [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return []


# ── Helper: find Boltz-2 output CIF files after a run ─────────────────────────
# Boltz-2 writes mmCIF files with names like:
#   {out_dir}/predictions/{stem}/model_0.cif, model_1.cif, ...
# We discover them in the aggregate rule via glob.


def boltz_output_cifs(protein, conformation):
    pattern = (
        BOLTZ_OUT / protein / conformation / "boltz_out" / "predictions" / "*" / "*.cif"
    )
    return sorted(Path(".").glob(str(pattern)))


def boltz_done_file(protein, conformation):
    return str(BOLTZ_OUT / protein / conformation / "prediction.done")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE CHECKPOINTS & RULE ALL
# ─────────────────────────────────────────────────────────────────────────────


checkpoint msa_checkpoint:
    """
Checkpoint: runs Stage 1 (MSA). After completion, Snakemake re-evaluates
the DAG using the discovered protein list.
"""
    output:
        sentinel=str(MSA_SENTINEL),
    log:
        str(LOG_DIR / "msa.log"),
    params:
        fasta_dir=str(FASTA_DIR),
        a3m_dir=str(MSA_DIR / "a3m"),
        pdb_dir=str(MSA_DIR / "pdb"),
        sh_dir=str(MSA_DIR / "sh"),
        query=config["uniprot"]["query"],
        size=config["uniprot"]["size"],
        delay=config["colabfold"]["delay"],
        retries=config["colabfold"]["max_retries"],
        poll_int=config["colabfold"]["poll_interval"],
        poll_timeout=config["colabfold"]["poll_timeout"],
    shell:
        """
        python scripts/run_msa.py \\
        --fasta-dir {params.fasta_dir} \\
        --a3m-dir {params.a3m_dir} \\
        --pdb-dir {params.pdb_dir} \\
        --sh-dir {params.sh_dir} \\
        --sentinel {output.sentinel} \\
        --query '{params.query}' \\
        --size {params.size} \\
        --delay {params.delay} \\
        --retries {params.retries} \\
        --poll-interval {params.poll_int} \\
        --poll-timeout {params.poll_timeout} \\
        >{log} 2>&1
        """


# ── Stage 2: InterProScan ─────────────────────────────────────────────────────


rule run_interproscan:
    input:
        fasta=str(FASTA_DIR / "npf_arabidopsis.fasta"),
        msa_done=str(MSA_SENTINEL),  # ensures FASTA is ready
    output:
        sentinel=str(IPRO_SENTINEL),
        summary=str(INTERPRO_DIR / "cdd_summary.json"),
    log:
        str(LOG_DIR / "interproscan.log"),
    params:
        out_dir=str(INTERPRO_DIR),
        email=config["interproscan"]["email"],
        accessions=" ".join(config["interproscan"]["cdd_accessions"]),
        poll_int=config["interproscan"]["poll_interval"],
        poll_timeout=config["interproscan"]["poll_timeout"],
    shell:
        """
        python scripts/run_interproscan.py \\
        --fasta {input.fasta} \\
        --out-dir {params.out_dir} \\
        --email {params.email} \\
        --accessions {params.accessions} \\
        --sentinel {output.sentinel} \\
        --poll-interval {params.poll_int} \\
        --poll-timeout {params.poll_timeout} \\
        >{log} 2>&1
        """


# ── Stage 2b: TM helix annotation (Phobius + TMHMM) ──────────────────────────
# Runs independently of CDD interproscan — results cached as {name}_tm.json.
# Outputs tm_topology_summary.json used by compute_tm_angle (Stage 11).


rule run_tm_annotation:
    input:
        fasta=str(FASTA_DIR / "npf_arabidopsis.fasta"),
        msa_done=str(MSA_SENTINEL),
    output:
        sentinel=str(TM_ANN_SENTINEL),
        topology=str(INTERPRO_DIR / "tm_topology_summary.json"),
    log:
        str(LOG_DIR / "tm_annotation.log"),
    params:
        out_dir=str(INTERPRO_DIR),
        email=TM_ANN_CFG["email"],
        poll_int=TM_ANN_CFG["poll_interval"],
        poll_timeout=TM_ANN_CFG["poll_timeout"],
    shell:
        """
        python scripts/run_tm_annotation.py \\
        --fasta {input.fasta} \\
        --out-dir {params.out_dir} \\
        --email {params.email} \\
        --sentinel {output.sentinel} \\
        --poll-interval {params.poll_int} \\
        --poll-timeout {params.poll_timeout} \\
        >{log} 2>&1
        """


# ── Stage 3: Template download ────────────────────────────────────────────────


rule download_templates:
    output:
        sentinel=str(TMPL_SENTINEL),
    log:
        str(LOG_DIR / "download_templates.log"),
    params:
        templates_root=str(TMPL_DIR),
    shell:
        """
        python scripts/download_templates.py \\
        --config config.yaml \\
        --templates-root {params.templates_root} \\
        --sentinel {output.sentinel} \\
        >{log} 2>&1
        """


# ── Stage 4: Prepare Boltz-2 input YAML ──────────────────────────────────────


rule prepare_boltz_input:
    input:
        fasta=str(FASTA_DIR / "{protein}.fasta"),
        a3m=str(MSA_DIR / "a3m" / "{protein}.a3m"),
        cdd_summary=str(INTERPRO_DIR / "cdd_summary.json"),  # declared output of run_interproscan
        tmpl_done=str(TMPL_SENTINEL),
    output:
        yaml=str(BOLTZ_IN_DIR / "{protein}" / "{conformation}" / "target.yaml"),
    log:
        str(LOG_DIR / "boltz_input" / "{protein}" / "{conformation}.log"),
    params:
        templates_dir=str(TMPL_DIR / "{conformation}"),
        ligand_smiles=BOLTZ_CFG["ligand_smiles"],
        ligand_entity_id=BOLTZ_CFG["ligand_entity_id"],
        protein_entity_id=BOLTZ_CFG["protein_entity_id"],
        pocket_max_distance=BOLTZ_CFG["pocket_max_distance"],
        pocket_force=str(BOLTZ_CFG["pocket_force"]).lower(),
    shell:
        """
        python scripts/make_boltz_input.py \\
        --fasta {input.fasta} \\
        --a3m {input.a3m} \\
        --cdd-summary {input.cdd_summary} \\
        --protein-name {wildcards.protein} \\
        --templates-dir {params.templates_dir} \\
        --conformation {wildcards.conformation} \\
        --output {output.yaml} \\
        --ligand-smiles '{params.ligand_smiles}' \\
        --ligand-entity-id {params.ligand_entity_id} \\
        --protein-entity-id {params.protein_entity_id} \\
        --pocket-max-distance {params.pocket_max_distance} \\
        --pocket-force {params.pocket_force} \\
        >{log} 2>&1
        """


# ── Stage 5: Run Boltz-2 ─────────────────────────────────────────────────────


rule run_boltz2:
    input:
        yaml=str(BOLTZ_IN_DIR / "{protein}" / "{conformation}" / "target.yaml"),
    output:
        done=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "prediction.done"),
    log:
        str(LOG_DIR / "boltz_run" / "{protein}" / "{conformation}.log"),
    conda:
        "envs/boltz2.yaml"
    params:
        out_dir=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "boltz_out"),
        recycling_steps=BOLTZ_CFG["recycling_steps"],
        diffusion_samples=BOLTZ_CFG["diffusion_samples"],
        output_format=BOLTZ_CFG["output_format"],
        accelerator=BOLTZ_CFG["accelerator"],
        no_kernels_flag="--no_kernels" if BOLTZ_CFG["no_kernels"] else "",
        extra_flags=BOLTZ_CFG.get("extra_flags", ""),
    shell:
        """
        boltz predict \\
        {input.yaml} \\
        --out_dir {params.out_dir} \\
        --recycling_steps {params.recycling_steps} \\
        --diffusion_samples {params.diffusion_samples} \\
        --output_format {params.output_format} \\
        --accelerator {params.accelerator} \\
        {params.no_kernels_flag} \\
        {params.extra_flags} \\
        >{log} 2>&1
        echo "$(date): prediction finished" >{output.done}
        """


# ── Stage 6: Discover Boltz-2 output CIFs (checkpoint) ───────────────────────
# After Boltz-2 runs, we need to discover the actual CIF filenames it wrote
# (they include a hash/timestamp in the path). A checkpoint lets Snakemake
# re-evaluate the DAG once the files exist.

# ── Stage 6a: Discover Boltz-2 output CIFs (checkpoint) ──────────────────────
# Boltz-2 writes CIF files under a content-hashed subdirectory whose name is
# not known until the prediction finishes. This checkpoint globs the output
# directory after run_boltz2 completes and records every CIF path in a JSON
# manifest so that downstream rules can expand the {sample_id} wildcard.
#
# Boltz-2 output layout (example, 5 diffusion samples):
#   results/boltz/{protein}/{conformation}/boltz_out/
#     predictions/
#       {protein}_target/          ← subfolder name matches input stem
#         model_0.cif
#         model_1.cif
#         model_2.cif
#         model_3.cif
#         model_4.cif
#
# {sample_id} = CIF stem, e.g. "model_0", "model_1", ...


checkpoint discover_boltz_outputs:
    input:
        done=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "prediction.done"),
    output:
        manifest=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "cif_manifest.json"),
    run:
        pred_root = (
            Path(BOLTZ_OUT)
            / wildcards.protein
            / wildcards.conformation
            / "boltz_out"
            / "predictions"
        )
        cifs = sorted(pred_root.glob("**/*.cif")) if pred_root.exists() else []
        if not cifs:
            raise RuntimeError(
                f"No CIF files found under {pred_root}. "
                "Check that run_boltz2 completed successfully."
            )
        manifest = [str(c) for c in cifs]
        Path(output.manifest).write_text(json.dumps(manifest, indent=2))
        print(
            f"[manifest] {wildcards.protein}/{wildcards.conformation}: "
            f"{len(manifest)} CIF(s) discovered"
        )


def _get_manifest(protein, conformation):
    """Return the parsed CIF manifest list for one protein × conformation."""
    cp = checkpoints.discover_boltz_outputs.get(
        protein=protein,
        conformation=conformation,
    )
    return json.loads(Path(cp.output.manifest).read_text())


def _sample_ids(protein, conformation):
    """Return the list of sample_id strings (CIF stems) for one run."""
    return [Path(c).stem for c in _get_manifest(protein, conformation)]


def _cif_for_sample(wildcards):
    """
    Input function for minimize_cif: return the actual CIF path that corresponds
    to this protein × conformation × sample_id combination.
    """
    cifs = _get_manifest(wildcards.protein, wildcards.conformation)
    for cif in cifs:
        if Path(cif).stem == wildcards.sample_id:
            return cif
    raise RuntimeError(
        f"CIF not found for sample_id='{wildcards.sample_id}' in "
        f"{wildcards.protein}/{wildcards.conformation}"
    )


# ── Stage 6b: ChimeraX minimization ──────────────────────────────────────────


rule minimize_cif:
    input:
        cif=_cif_for_sample,
    output:
        pdb=str(
            MIN_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized.pdb"
        ),
    log:
        str(LOG_DIR / "minimize" / "{protein}" / "{conformation}" / "{sample_id}.log"),
    params:
        chimerax=CHIMERAX_BIN,
    shell:
        """
        mkdir -p $(dirname {output.pdb})
        {params.chimerax} --nogui \\
        --script "scripts/minimize_cif.py {input.cif} {output.pdb}" \\
        >{log} 2>&1
        """


# ── Stage 7: PLIP via Docker ──────────────────────────────────────────────────


rule run_plip:
    input:
        pdb=str(
            MIN_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized.pdb"
        ),
    output:
        report=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "model_minimized_report.txt"
        ),
    log:
        str(LOG_DIR / "plip" / "{protein}" / "{conformation}" / "{sample_id}.log"),
    params:
        image=PLIP_CFG["image"],
        docker_memory=PLIP_CFG["docker_memory"],
        platform_flag=(
            f"--platform {PLIP_CFG['docker_platform']}"
            if PLIP_CFG.get("docker_platform")
            else ""
        ),
        chains_flag=(
            f'--chains "[[\\"{PLIP_CFG["receptor_chain"]}\\","'
            f'\\"{PLIP_CFG["ligand_chain"]}\\""]]"'
        ),
        report_dir=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
        ),
    shell:
        """
        mkdir -p {params.report_dir}
        docker run --rm \\
        --memory={params.docker_memory} \\
        {params.platform_flag} \\
        -v $(pwd):/work -w /work \\
        {params.image} \\
        -f {input.pdb} \\
        -t \\
        {params.chains_flag} \\
        -o {params.report_dir} \\
        >{log} 2>&1
        """


# ── Stage 8: pliparser — PLIP report → CSV + CXC ─────────────────────────────


rule plip_to_csv:
    input:
        report=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "model_minimized_report.txt"
        ),
    output:
        summary=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "csv"
            / "summary.csv"
        ),
    log:
        str(
            LOG_DIR
            / "pliparser"
            / "{protein}"
            / "{conformation}"
            / "{sample_id}_csv.log"
        ),
    conda:
        "envs/pliparser.yaml"
    params:
        out_dir=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "csv"
        ),
    shell:
        """
        mkdir -p {params.out_dir}
        pliparser plip2csv \\
        --input {input.report} \\
        --output {params.out_dir}/ \\
        >{log} 2>&1
        """


rule csv_to_cxc:
    input:
        csv_dir=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "csv"
        ),
        pdb=str(
            MIN_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized.pdb"
        ),
    output:
        cxc=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "interaction.cxc"
        ),
        config_json=str(
            PLIP_DIR
            / "{protein}"
            / "{conformation}"
            / "{sample_id}"
            / "model_minimized_report"
            / "cxc-config.json"
        ),
    log:
        str(
            LOG_DIR
            / "pliparser"
            / "{protein}"
            / "{conformation}"
            / "{sample_id}_cxc.log"
        ),
    conda:
        "envs/pliparser.yaml"
    params:
        receptor_chain=PLIP_CFG["receptor_chain"],
        ligand_chain=PLIP_CFG["ligand_chain"],
        transparency=VIZ_CFG["transparency"],
        receptor_color=VIZ_CFG["receptor_color"],
        ligand_color=VIZ_CFG["ligand_color"],
    shell:
        """
        python scripts/make_cxc_config.py \\
        --pdb {input.pdb} \\
        --output {output.config_json} \\
        --receptor-chain {params.receptor_chain} \\
        --ligand-chain {params.ligand_chain} \\
        --transparency {params.transparency} \\
        --receptor-color {params.receptor_color} \\
        --ligand-color {params.ligand_color} \\
        >{log} 2>&1

        pliparser csv2cxc \\
        --input {input.csv_dir} \\
        --output {output.cxc} \\
        --config {output.config_json} \\
        >>{log} 2>&1
        """


# ── Stage 9: Aggregate per protein × conformation ────────────────────────────


def summaries_for_protein_conformation(wildcards):
    """
    Expand the list of per-sample summary CSVs for one protein × conformation.
    Called after discover_boltz_outputs checkpoint has written the manifest,
    so sample_ids are known and the full list of expected CSVs can be returned.
    """
    sample_ids = _sample_ids(wildcards.protein, wildcards.conformation)
    return [
        str(
            PLIP_DIR
            / wildcards.protein
            / wildcards.conformation
            / sid
            / "model_minimized_report"
            / "csv"
            / "summary.csv"
        )
        for sid in sample_ids
    ]


rule aggregate_plip:
    input:
        csvs=summaries_for_protein_conformation,
    output:
        str(PLIP_DIR / "{protein}" / "{conformation}" / "summary.csv"),
    log:
        str(LOG_DIR / "aggregate" / "{protein}" / "{conformation}.log"),
    conda:
        "envs/aggregate.yaml"
    shell:
        """
        python scripts/aggregate_plip_summary.py \\
        --output {output} \\
        {input.csvs} \\
        >{log} 2>&1
        """


# ─────────────────────────────────────────────────────────────────────────────
# STAGES 10–13 — Conformational angle classification via GMM
# ─────────────────────────────────────────────────────────────────────────────
# Dependency graph (per protein × conformation, batched across all sample_ids):
#
#   discover_boltz_outputs ─┬─► minimize_cif ─► run_plip ─► ... (existing chain)
#                           │
#                           └─► run_dssp_batch ────────────────────────────────┐
#                                (one ChimeraX session, all samples)           ▼
#                           run_tm_annotation ──► compute_tm_angles_batch ─► angles.csv
#                                (one Python call, all samples)
#                                                          │
#                                                          ▼
#                                                    gmm_analysis (global)
#
# Packaging per protein × conformation reduces ChimeraX launch overhead from
# N_samples jobs to 1 job per protein × conformation.


# ── Stage 10: ChimeraX DSSP on raw Boltz-2 CIFs (batched per protein×conf) ───
# One ChimeraX session opens all samples for a protein × conformation in sequence,
# writes results/dssp/{protein}/{conformation}/{sample_id}/dssp.json for each,
# and touches a sentinel when done.


rule run_dssp_batch:
    input:
        manifest=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "cif_manifest.json"),
    output:
        sentinel=str(DSSP_DIR / "{protein}" / "{conformation}" / "dssp.done"),
    log:
        str(LOG_DIR / "dssp" / "{protein}" / "{conformation}.log"),
    params:
        chimerax=CHIMERAX_BIN,
        dssp_dir=str(DSSP_DIR / "{protein}" / "{conformation}"),
    shell:
        """
        mkdir -p {params.dssp_dir}
        {params.chimerax} --nogui \\
        --script "scripts/chimerax_dssp.py {input.manifest} {params.dssp_dir}" \\
        >{log} 2>&1
        touch {output.sentinel}
        """


# ── Stages 11+12: TM2/TM8 angle for all samples → angles.csv (batched) ───────
# One Python call per protein × conformation processes all samples and writes
# angles.csv directly, replacing both the per-sample compute_tm_angle rule and
# the separate collect_tm_angles aggregation step.


rule compute_tm_angles_batch:
    input:
        manifest=str(BOLTZ_OUT / "{protein}" / "{conformation}" / "cif_manifest.json"),
        dssp_done=str(DSSP_DIR / "{protein}" / "{conformation}" / "dssp.done"),
        topology=str(INTERPRO_DIR / "tm_topology_summary.json"),
    output:
        angles_csv=str(TM_ANG_DIR / "{protein}" / "{conformation}" / "angles.csv"),
    log:
        str(LOG_DIR / "tm_angle" / "{protein}" / "{conformation}.log"),
    conda:
        "envs/tm_analysis.yaml"
    params:
        dssp_dir=str(DSSP_DIR / "{protein}" / "{conformation}"),
        min_helix_frac=TM_ANG_CFG["min_helix_frac"],
    shell:
        """
        python scripts/compute_tm_angle.py \\
        --batch-manifest {input.manifest} \\
        --dssp-dir {params.dssp_dir} \\
        --topology {input.topology} \\
        --protein {wildcards.protein} \\
        --conformation {wildcards.conformation} \\
        --output {output.angles_csv} \\
        --min-helix-frac {params.min_helix_frac} \\
        >{log} 2>&1
        """


# ── Stage 13: Fit GMM-3 and GMM-6 + BIC comparison (global) ──────────────────


def all_angle_csv_inputs(wildcards):
    """
    Return all per-conformation angle CSVs (one per protein × conformation).
    Called after all discover_boltz_outputs checkpoints have fired.
    """
    sentinel = checkpoints.msa_checkpoint.get().output.sentinel
    proteins = read_proteins(sentinel)
    targets = []
    for protein in proteins:
        for conformation in CONFORMATIONS:
            _sample_ids(protein, conformation)  # ensures checkpoint is resolved
            targets.append(str(TM_ANG_DIR / protein / conformation / "angles.csv"))
    return targets


rule gmm_analysis:
    input:
        csvs=all_angle_csv_inputs,
    output:
        sentinel=str(GMM_DIR / "gmm.done"),
        report=str(GMM_DIR / "gmm_report.json"),
    log:
        str(LOG_DIR / "gmm_analysis.log"),
    conda:
        "envs/tm_analysis.yaml"
    params:
        out_dir=str(GMM_DIR),
        n_init=GMM_CFG["n_init"],
    shell:
        """
        python scripts/gmm_conformation.py \\
        --input {input.csvs} \\
        --output-dir {params.out_dir} \\
        --sentinel {output.sentinel} \\
        --n-init {params.n_init} \\
        >{log} 2>&1
        """


# ─────────────────────────────────────────────────────────────────────────────
# RULE ALL — top-level target
# ─────────────────────────────────────────────────────────────────────────────
# Snakemake checkpoint pattern (Snakemake 8):
#
#   When an input function calls checkpoints.X.get() and the checkpoint has
#   NOT yet run, Snakemake raises IncompleteCheckpointException internally.
#   This exception MUST propagate uncaught — that is the signal Snakemake
#   uses to know it must execute the checkpoint first, then re-evaluate.
#   Catching it causes Snakemake to see an empty list and exit immediately.
#
#   Two-checkpoint cascade:
#     1. msa_checkpoint            → produces the protein list
#     2. discover_boltz_outputs    → produces sample_ids per protein×conformation


def all_final_outputs(wildcards):
    """
    Return all final outputs:
      - PLIP aggregate CSVs (one per protein × conformation)
      - GMM analysis report (global)

    IncompleteCheckpointException must NOT be caught here — Snakemake
    uses it to schedule the required checkpoint and retry automatically.
    """
    # Raises IncompleteCheckpointException if MSA not done yet → triggers msa_checkpoint
    sentinel = checkpoints.msa_checkpoint.get().output.sentinel
    proteins = read_proteins(sentinel)

    targets = []
    for protein in proteins:
        for conformation in CONFORMATIONS:
            # Raises IncompleteCheckpointException if Boltz-2 not done yet
            # → triggers discover_boltz_outputs for this protein × conformation
            _sample_ids(protein, conformation)

            targets.append(str(PLIP_DIR / protein / conformation / "summary.csv"))

    # GMM report depends on all per-conformation angle CSVs (collected above)
    targets.append(str(GMM_DIR / "gmm_report.json"))

    return targets


rule all:
    default_target: True
    input:
        all_final_outputs,
