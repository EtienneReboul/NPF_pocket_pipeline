#!/usr/bin/env python3
"""
make_cxc_config.py
==================
Generates the JSON config file required by pliparser csv2cxc --config.
 
Usage (called by Snakemake, but also works standalone):
    python scripts/make_cxc_config.py \
        --pdb       results/.../model_0_protonated.pdb \
        --output    results/.../cxc-config.json \
        --receptor-chain A \
        --ligand-chain   B \
        --transparency   65 \
        --receptor-color gray \
        --ligand-color   green
"""
 
import argparse
import json
from pathlib import Path
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a pliparser csv2cxc JSON config file."
    )
    parser.add_argument("--pdb",             required=True,  help="Absolute path to the protonated PDB.")
    parser.add_argument("--output",          required=True,  help="Path for the output JSON config.")
    parser.add_argument("--model-id",        type=int, default=1)
    parser.add_argument("--receptor-chain",  required=True)
    parser.add_argument("--ligand-chain",    required=True)
    parser.add_argument("--transparency",    type=int, default=65)
    parser.add_argument("--issmalmol",       action="store_true",
                        help="Set to true when the ligand is a small molecule within the receptor chain.")
    parser.add_argument("--receptor-color",  default="gray")
    parser.add_argument("--ligand-color",    default="green")
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    config = {
        "pdb":            str(Path(args.pdb).resolve()),
        "model_id":       args.model_id,
        "receptor_chain": args.receptor_chain,
        "ligand_chain":   args.ligand_chain,
        "transparency":   args.transparency,
        "issmalmol":      args.issmalmol,
        "receptor_color": args.receptor_color,
        "ligand_color":   args.ligand_color,
    }
 
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(config, fh, indent=4)
 
    print(f"[OK] Config written to {out}")
 
 
if __name__ == "__main__":
    main()