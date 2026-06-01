"""
NPF Family Structure Modelling Pipeline — Master Snakefile
==========================================================

Stages
------
  1. msa            — UniProt download + ColabFold MSA per NPF member
  2. interproscan   — EMBL-EBI CDD annotation + binding-site residue extraction
  3. templates      — RCSB mmCIF download per conformation set
  4. boltz_input    — generate Boltz-2 YAML per protein × conformation
  5. boltz_run      — run Boltz-2 per protein × conformation
  6. minimize       — ChimeraX energy minimization, mmCIF → PDB per diffusion sample
  7. plip           — PLIP via Docker per sample
  8. pliparser      — PLIP report → CSV + ChimeraX .cxc visualisation
  9. aggregate      — merge per-sample CSVs into one CSV per protein × conformation

Output tree
-----------
  results/
    boltz/{protein}/{conformation}/
      boltz_out/                       ← Boltz-2 mmCIF outputs
    minimized/{protein}/{conformation}/
      sample_{N}/
        model_minimized.pdb
        model_minimized_energy.csv
    plip/{protein}/{conformation}/
      sample_{N}/
        {model_id}_report/
          {model_id}_report.txt
          {model_id}_protonated.pdb
          csv/summary.csv
          {model_id}.cxc
      summary.csv                      ← ★ aggregated across all samples
"""

import json
import re
from pathlib import Path
from collections import defaultdict

configfile: "config.yaml"

# ── Directory shortcuts ────────────────────────────────────────────────────────
DIRS         = config["dirs"]
FASTA_DIR    = Path(DIRS["fasta"])
MSA_DIR      = Path(DIRS["msa"])
INTERPRO_DIR = Path(DIRS["interpro"])
TMPL_DIR     = Path(DIRS["templates"])
BOLTZ_IN_DIR = Path(DIRS["boltz_in"])
BOLTZ_OUT    = Path(DIRS["boltz_out"])
MIN_DIR      = Path(DIRS["minimized"])
PLIP_DIR     = Path(DIRS["plip"])
LOG_DIR      = Path(DIRS["logs"])

# ── Config shortcuts ───────────────────────────────────────────────────────────
BOLTZ_CFG    = config["boltz"]
PLIP_CFG     = config["plip"]
VIZ_CFG      = config["visualization"]
CHIMERAX_BIN = config["chimerax_bin"]
CONFORMATIONS = list(config["templates"]["conformations"].keys())

# ── Lazy wildcard discovery ────────────────────────────────────────────────────
# Proteins are discovered after Stage 1 completes (sentinel file).
# The sentinel lists one protein base-name per line.
# We use a checkpoint so Snakemake re-evaluates the DAG after MSA finishes.

MSA_SENTINEL   = MSA_DIR / "msa.done"
IPRO_SENTINEL  = INTERPRO_DIR / "interproscan.done"
TMPL_SENTINEL  = TMPL_DIR / "templates.done"

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
    pattern = BOLTZ_OUT / protein / conformation / "boltz_out" / "predictions" / "*" / "*.cif"
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
        sentinel = str(MSA_SENTINEL),
    params:
        fasta_dir    = str(FASTA_DIR),
        a3m_dir      = str(MSA_DIR / "a3m"),
        pdb_dir      = str(MSA_DIR / "pdb"),
        sh_dir       = str(MSA_DIR / "sh"),
        query        = config["uniprot"]["query"],
        size         = config["uniprot"]["size"],
        delay        = config["colabfold"]["delay"],
        retries      = config["colabfold"]["max_retries"],
        poll_int     = config["colabfold"]["poll_interval"],
        poll_timeout = config["colabfold"]["poll_timeout"],
    log:
        str(LOG_DIR / "msa.log"),
    shell:
        """
        python scripts/run_msa.py \\
            --fasta-dir     {params.fasta_dir} \\
            --a3m-dir       {params.a3m_dir} \\
            --pdb-dir       {params.pdb_dir} \\
            --sh-dir        {params.sh_dir} \\
            --sentinel      {output.sentinel} \\
            --query         '{params.query}' \\
            --size          {params.size} \\
            --delay         {params.delay} \\
            --retries       {params.retries} \\
            --poll-interval {params.poll_int} \\
            --poll-timeout  {params.poll_timeout} \\
        > {log} 2>&1
        """


# ── Stage 2: InterProScan ─────────────────────────────────────────────────────

rule run_interproscan:
    input:
        fasta    = str(FASTA_DIR / "npf_arabidopsis.fasta"),
        msa_done = str(MSA_SENTINEL),   # ensures FASTA is ready
    output:
        sentinel = str(IPRO_SENTINEL),
        json     = str(INTERPRO_DIR / "interproscan_cdd.json"),
        summary  = str(INTERPRO_DIR / "cdd_summary.json"),
    params:
        out_dir      = str(INTERPRO_DIR),
        email        = config["interproscan"]["email"],
        accessions   = " ".join(config["interproscan"]["cdd_accessions"]),
        poll_int     = config["interproscan"]["poll_interval"],
        poll_timeout = config["interproscan"]["poll_timeout"],
    log:
        str(LOG_DIR / "interproscan.log"),
    shell:
        """
        python scripts/run_interproscan.py \\
            --fasta         {input.fasta} \\
            --out-dir       {params.out_dir} \\
            --email         {params.email} \\
            --accessions    {params.accessions} \\
            --sentinel      {output.sentinel} \\
            --poll-interval {params.poll_int} \\
            --poll-timeout  {params.poll_timeout} \\
        > {log} 2>&1
        """


# ── Stage 3: Template download ────────────────────────────────────────────────

rule download_templates:
    output:
        sentinel = str(TMPL_SENTINEL),
    params:
        templates_root = str(TMPL_DIR),
    log:
        str(LOG_DIR / "download_templates.log"),
    shell:
        """
        python scripts/download_templates.py \\
            --config         config.yaml \\
            --templates-root {params.templates_root} \\
            --sentinel       {output.sentinel} \\
        > {log} 2>&1
        """


# ── Stage 4: Prepare Boltz-2 input YAML ──────────────────────────────────────

rule prepare_boltz_input:
    input:
        fasta     = str(FASTA_DIR / "{protein}.fasta"),
        a3m       = str(MSA_DIR / "a3m" / "{protein}.a3m"),
        residues  = str(INTERPRO_DIR / "{protein}_binding_site_residues.txt"),
        tmpl_done = str(TMPL_SENTINEL),
    output:
        yaml = str(BOLTZ_IN_DIR / "{protein}" / "{conformation}" / "target.yaml"),
    params:
        templates_dir       = str(TMPL_DIR / "{conformation}"),
        ligand_smiles       = BOLTZ_CFG["ligand_smiles"],
        ligand_entity_id    = BOLTZ_CFG["ligand_entity_id"],
        protein_entity_id   = BOLTZ_CFG["protein_entity_id"],
        pocket_max_distance = BOLTZ_CFG["pocket_max_distance"],
        pocket_force        = str(BOLTZ_CFG["pocket_force"]).lower(),
    log:
        str(LOG_DIR / "boltz_input" / "{protein}" / "{conformation}.log"),
    shell:
        """
        python scripts/make_boltz_input.py \\
            --fasta               {input.fasta} \\
            --a3m                 {input.a3m} \\
            --residues-file       {input.residues} \\
            --templates-dir       {params.templates_dir} \\
            --conformation        {wildcards.conformation} \\
            --output              {output.yaml} \\
            --ligand-smiles       '{params.ligand_smiles}' \\
            --ligand-entity-id    {params.ligand_entity_id} \\
            --protein-entity-id   {params.protein_entity_id} \\
            --pocket-max-distance {params.pocket_max_distance} \\
            --pocket-force        {params.pocket_force} \\
        > {log} 2>&1
        """


# ── Stage 5: Run Boltz-2 ─────────────────────────────────────────────────────

rule run_boltz2:
    input:
        yaml = str(BOLTZ_IN_DIR / "{protein}" / "{conformation}" / "target.yaml"),
    output:
        done = str(BOLTZ_OUT / "{protein}" / "{conformation}" / "prediction.done"),
    params:
        out_dir          = str(BOLTZ_OUT / "{protein}" / "{conformation}" / "boltz_out"),
        recycling_steps  = BOLTZ_CFG["recycling_steps"],
        diffusion_samples = BOLTZ_CFG["diffusion_samples"],
        output_format    = BOLTZ_CFG["output_format"],
        accelerator      = BOLTZ_CFG["accelerator"],
        no_kernels_flag  = "--no_kernels" if BOLTZ_CFG["no_kernels"] else "",
        extra_flags      = BOLTZ_CFG.get("extra_flags", ""),
    log:
        str(LOG_DIR / "boltz_run" / "{protein}" / "{conformation}.log"),
    shell:
        """
        boltz predict \\
            {input.yaml} \\
            --out_dir          {params.out_dir} \\
            --recycling_steps  {params.recycling_steps} \\
            --diffusion_samples {params.diffusion_samples} \\
            --output_format    {params.output_format} \\
            --accelerator      {params.accelerator} \\
            {params.no_kernels_flag} \\
            {params.extra_flags} \\
        > {log} 2>&1
        echo "$(date): prediction finished" > {output.done}
        """


# ── Stage 6: Discover Boltz-2 output CIFs (checkpoint) ───────────────────────
# After Boltz-2 runs, we need to discover the actual CIF filenames it wrote
# (they include a hash/timestamp in the path). A checkpoint lets Snakemake
# re-evaluate the DAG once the files exist.

checkpoint discover_boltz_outputs:
    """
    Triggered after run_boltz2 completes. Writes a JSON listing all CIF paths
    so downstream rules can glob them deterministically.
    """
    input:
        done = str(BOLTZ_OUT / "{protein}" / "{conformation}" / "prediction.done"),
    output:
        manifest = str(BOLTZ_OUT / "{protein}" / "{conformation}" / "cif_manifest.json"),
    run:
        pred_root = Path(BOLTZ_OUT / wildcards.protein / wildcards.conformation
                         / "boltz_out" / "predictions")
        cifs = sorted(pred_root.glob("**/*.cif")) if pred_root.exists() else []
        manifest = [str(c) for c in cifs]
        Path(output.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"[manifest] {wildcards.protein}/{wildcards.conformation}: {len(manifest)} CIFs")


def get_cifs_from_manifest(wildcards):
    manifest_path = checkpoints.discover_boltz_outputs.get(
        protein=wildcards.protein,
        conformation=wildcards.conformation,
    ).output.manifest
    return json.loads(Path(manifest_path).read_text())


def sample_id_from_cif(cif_path: str) -> str:
    """
    Derive a stable sample ID from the CIF filename.
    Boltz-2 writes model_0.cif, model_1.cif, ... inside a prediction sub-folder.
    """
    return Path(cif_path).stem   # e.g. "model_0"


# ── Stage 6: ChimeraX minimization ───────────────────────────────────────────

rule minimize_cif:
    input:
        cif = lambda wc: str(wc.cif_path),
    output:
        pdb = str(MIN_DIR / "{protein}" / "{conformation}" / "{sample_id}" / "model_minimized.pdb"),
    params:
        chimerax = CHIMERAX_BIN,
    log:
        str(LOG_DIR / "minimize" / "{protein}" / "{conformation}" / "{sample_id}.log"),
    shell:
        """
        mkdir -p $(dirname {output.pdb})
        {params.chimerax} --nogui \\
            --script "scripts/minimize_cif.py {input.cif} {output.pdb}" \\
        > {log} 2>&1
        """


# ── Stage 7: PLIP via Docker ──────────────────────────────────────────────────

rule run_plip:
    input:
        pdb = str(MIN_DIR / "{protein}" / "{conformation}" / "{sample_id}" / "model_minimized.pdb"),
    output:
        report = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "model_minimized_report.txt"
        ),
    params:
        image             = PLIP_CFG["image"],
        docker_memory     = PLIP_CFG["docker_memory"],
        platform_flag     = f"--platform {PLIP_CFG['docker_platform']}" if PLIP_CFG.get("docker_platform") else "",
        receptor_chain    = PLIP_CFG["receptor_chain"],
        ligand_chain      = PLIP_CFG["ligand_chain"],
        chains_flag       = f'--chains "[[\\"{PLIP_CFG["receptor_chain"]}\\",\\"{PLIP_CFG["ligand_chain"]}\\""]]"',
        report_dir        = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report"
        ),
    log:
        str(LOG_DIR / "plip" / "{protein}" / "{conformation}" / "{sample_id}.log"),
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
        > {log} 2>&1
        """


# ── Stage 8: pliparser — PLIP report → CSV + CXC ─────────────────────────────

rule plip_to_csv:
    input:
        report = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "model_minimized_report.txt"
        ),
    output:
        summary = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "csv" / "summary.csv"
        ),
    params:
        out_dir = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "csv"
        ),
    log:
        str(LOG_DIR / "pliparser" / "{protein}" / "{conformation}" / "{sample_id}_csv.log"),
    conda:
        "envs/pliparser.yaml"
    shell:
        """
        mkdir -p {params.out_dir}
        pliparser plip2csv \\
            --input  {input.report} \\
            --output {params.out_dir}/ \\
        > {log} 2>&1
        """


rule csv_to_cxc:
    input:
        csv_dir = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "csv"
        ),
        pdb = str(
            MIN_DIR / "{protein}" / "{conformation}" / "{sample_id}" / "model_minimized.pdb"
        ),
    output:
        cxc    = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "interaction.cxc"
        ),
        config = str(
            PLIP_DIR / "{protein}" / "{conformation}" / "{sample_id}"
            / "model_minimized_report" / "cxc-config.json"
        ),
    params:
        receptor_chain = PLIP_CFG["receptor_chain"],
        ligand_chain   = PLIP_CFG["ligand_chain"],
        transparency   = VIZ_CFG["transparency"],
        receptor_color = VIZ_CFG["receptor_color"],
        ligand_color   = VIZ_CFG["ligand_color"],
    log:
        str(LOG_DIR / "pliparser" / "{protein}" / "{conformation}" / "{sample_id}_cxc.log"),
    conda:
        "envs/pliparser.yaml"
    shell:
        """
        python scripts/make_cxc_config.py \\
            --pdb             {input.pdb} \\
            --output          {output.config} \\
            --receptor-chain  {params.receptor_chain} \\
            --ligand-chain    {params.ligand_chain} \\
            --transparency    {params.transparency} \\
            --receptor-color  {params.receptor_color} \\
            --ligand-color    {params.ligand_color} \\
        >> {log} 2>&1

        pliparser csv2cxc \\
            --input  {input.csv_dir} \\
            --output {output.cxc} \\
            --config {output.config} \\
        >> {log} 2>&1
        """


# ── Stage 9: Aggregate per protein × conformation ────────────────────────────

def summaries_for_protein_conformation(wildcards):
    """
    Collect all per-sample summary CSVs for one protein × conformation.
    Requires the discover_boltz_outputs checkpoint to have run.
    """
    manifest_path = checkpoints.discover_boltz_outputs.get(
        protein=wildcards.protein,
        conformation=wildcards.conformation,
    ).output.manifest
    cifs = json.loads(Path(manifest_path).read_text())
    return [
        str(PLIP_DIR / wildcards.protein / wildcards.conformation
            / Path(c).stem / "model_minimized_report" / "csv" / "summary.csv")
        for c in cifs
    ]


rule aggregate_plip:
    input:
        csvs = summaries_for_protein_conformation,
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
        > {log} 2>&1
        """


# ─────────────────────────────────────────────────────────────────────────────
# RULE ALL — top-level target
# ─────────────────────────────────────────────────────────────────────────────

def all_final_outputs(wildcards):
    """
    Compute all expected final outputs once the MSA sentinel is available.
    Returns: list of aggregate summary CSV paths (one per protein × conformation).
    """
    proteins = read_proteins(checkpoints.msa_checkpoint.get().output.sentinel)
    targets  = []
    for protein in proteins:
        for conformation in CONFORMATIONS:
            targets.append(
                str(PLIP_DIR / protein / conformation / "summary.csv")
            )
    return targets


rule all:
    input:
        all_final_outputs,
