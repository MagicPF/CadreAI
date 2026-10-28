#!/usr/bin/env python3
"""Standalone Boltz-2 protein–ligand prediction, extracted from BindingSimulatorPro.

Inputs: protein sequence (literal or FASTA) and ligand SMILES (literal or text file).
Outputs: Boltz's native predictions plus a machine-readable result.json.

This script reports Boltz's affinity_pred_value as-is. It does not convert it
into experimentally measured affinity or docking energy.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from rdkit import Chem


def read_sequence(value: str) -> str:
    path = Path(value)
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        value = "".join(line.strip() for line in lines if not line.startswith(">"))
    sequence = re.sub(r"\s+", "", value).upper()
    if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
        raise ValueError("Protein sequence must contain amino-acid letters only")
    return sequence


def read_smiles(value: str) -> str:
    path = Path(value)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip().splitlines()[0].split()[0]
    molecule = Chem.MolFromSmiles(value.strip())
    if molecule is None:
        raise ValueError("Invalid ligand SMILES")
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule, canonical=True)


def collect_json(output_dir: Path) -> dict:
    files = sorted(output_dir.rglob("*.json"))
    results = {}
    affinity_values = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for key, val in obj.items():
                walk(val, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, list):
            for index, val in enumerate(obj):
                walk(val, f"{prefix}[{index}]")
        elif "affinity_pred_value" in prefix and isinstance(obj, (int, float)) and not isinstance(obj, bool):
            affinity_values.append({"key": prefix, "value": float(obj)})

    for file in files:
        if file.name == "result.json":
            continue
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results[str(file.relative_to(output_dir))] = data
        walk(data, str(file.relative_to(output_dir)))
    return {"json_outputs": results, "affinity_predictions": affinity_values}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Boltz-2 prediction for a protein sequence and ligand SMILES")
    parser.add_argument("--sequence", required=True, help="Protein sequence or FASTA file")
    parser.add_argument("--smiles", required=True, help="SMILES string or SMILES text file")
    parser.add_argument("--run-name", default="boltz_run")
    parser.add_argument("--out-dir", default="boltz_output")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-msa-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow reuse of an existing run directory")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        parser.error("--run-name must be a simple filename (letters, numbers, underscore, dot, hyphen)")

    sequence = read_sequence(args.sequence)
    smiles = read_smiles(args.smiles)
    run_dir = Path(args.out_dir).resolve() / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        parser.error(f"Output already exists: {run_dir}. Choose another --run-name or use --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_yaml = run_dir / "input.yaml"
    config = {
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"receptor": ["A"], "binder": "B"}}],
    }
    input_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    command = ["boltz", "predict", str(input_yaml), "--out_dir", str(run_dir),
               "--accelerator", args.accelerator, "--seed", str(args.seed)]
    if args.use_msa_server:
        command.append("--use_msa_server")

    env = os.environ.copy()
    env.update({"MKL_THREADING_LAYER": "GNU", "MKL_SERVICE_FORCE_INTEL": "1",
                "PYTHONHASHSEED": str(args.seed), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    print("Command:", " ".join(command), flush=True)
    log_path = run_dir / "boltz.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError("The 'boltz' executable is not available on PATH") from exc

    structures = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*.cif"))
    if not structures:
        structures = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*.pdb"))
    prediction = collect_json(run_dir)
    result = {
        "run_name": args.run_name,
        "status": "success" if process.returncode == 0 and structures else "failed",
        "return_code": process.returncode,
        "seed": args.seed,
        "use_msa_server": args.use_msa_server,
        "protein_sequence": sequence,
        "ligand_smiles": smiles,
        "structure_files": structures,
        "affinity_predictions": prediction["affinity_predictions"],
        "json_output_files": list(prediction["json_outputs"]),
        "log_file": "boltz.log",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "structures": structures,
                      "affinity_predictions": result["affinity_predictions"],
                      "result_file": str(run_dir / "result.json")}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
