#!/usr/bin/env python3
"""CrossDocked2020 unconditional target-level Top-1 evaluation.

Reproduces the selection policy of monitor.py, but writes BOTH independent
Top-1 datasets and a transparent evaluation report. No hit-based prefilter.

Examples:
    python evaluate.py --root CadreAI --output-dir evaluation
    python evaluate.py --root CadreAI --allow-incomplete
    python evaluate.py --root CadreAI --strict-duplicates

Requires: RDKit. Input columns: SMILES, vina, qed, sa, composite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.error")
METRICS = ("vina", "qed", "sa", "composite")
EXPORT_FIELDS = ("task_id", "SMILES", "vina", "qed", "sa", "composite", "mol_weight",
                 "heavy_atom_count", "ligand_efficiency", "hit", "source_file")
THRESHOLDS = {"vina": -8.18, "qed": 0.25, "sa": 0.59}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("CadreAI"),
                   help="Experiment root containing cache/<task_id>/ and <task_id>/")
    p.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    p.add_argument("--total-targets", type=int, default=100)
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Permit incomplete target coverage (HR denominator remains total-targets)")
    p.add_argument("--strict-duplicates", action="store_true",
                   help="Fail when the same SMILES has conflicting metric values")
    p.add_argument("--duplicate-tolerance", type=float, default=1e-6)
    p.add_argument("--strict-composite", action="store_true",
                   help="Fail if stored composite disagrees with 0.5*QED-0.5*Vina+0.4*SA")
    return p.parse_args()


def sources(root: Path, task_id: int):
    tid = str(task_id)
    return (root / tid / "molecules.csv",
            root / "cache" / tid / "molecules.csv",
            root / "cache" / tid / "molecules_pool.csv")


def parse_record(row, file: Path, line: int, task_id: int):
    raw = (row.get("SMILES") or row.get("smiles") or "").strip()
    if not raw:
        raise ValueError(f"Missing SMILES at {file}:{line}")
    mol = Chem.MolFromSmiles(raw)
    if mol is None or mol.GetNumHeavyAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Invalid/split SMILES at {file}:{line}: {raw}")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    values = [(row.get(key) or "").strip() for key in METRICS]
    if not all(values):
        return canonical, None  # monitor.py counts the molecule but skips records without metrics
    try:
        metrics = dict(zip(METRICS, (float(v) for v in values)))
    except ValueError as exc:
        raise ValueError(f"Nonnumeric metrics at {file}:{line}") from exc
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError(f"Nonfinite metrics at {file}:{line}")
    if metrics["vina"] >= 500:  # documented docking failure sentinel
        return canonical, None
    hac = mol.GetNumHeavyAtoms()
    record = dict(task_id=task_id, SMILES=canonical, **metrics,
                  mol_weight=Descriptors.MolWt(mol), heavy_atom_count=hac,
                  ligand_efficiency=-metrics["vina"] / hac,
                  source_file=str(file))
    return canonical, record


def read_target(root, task_id, args, warnings):
    unique_smiles = set()
    records = {}
    files = 0
    metric_rows = 0
    conflicts = 0
    composite_mismatches = 0
    for file in sources(root, task_id):
        if not file.is_file():
            continue
        files += 1
        with file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not ({"SMILES", "smiles"} & set(reader.fieldnames)):
                raise ValueError(f"Missing SMILES column: {file}")
            for line, row in enumerate(reader, 2):
                canonical, record = parse_record(row, file, line, task_id)
                unique_smiles.add(canonical)
                if record is None:
                    continue
                metric_rows += 1
                expected = 0.5 * record["qed"] - 0.5 * record["vina"] + 0.4 * record["sa"]
                if abs(record["composite"] - expected) > 1e-3:
                    composite_mismatches += 1
                    if args.strict_composite:
                        raise ValueError(f"Composite mismatch: {file}:{line}, got {record['composite']}, expected {expected}")
                previous = records.get(canonical)
                if previous is not None:
                    differing = any(abs(record[k] - previous[k]) > args.duplicate_tolerance for k in METRICS)
                    if differing:
                        conflicts += 1
                        if args.strict_duplicates:
                            raise ValueError(f"Conflicting metrics for target {task_id} SMILES {canonical}: "
                                             f"{previous['source_file']} vs {file}:{line}")
                    # Same behavior as monitor.py: for repeated SMILES, retain lowest Vina.
                    if (record["vina"], record["mol_weight"]) < (previous["vina"], previous["mol_weight"]):
                        records[canonical] = record
                else:
                    records[canonical] = record
    if conflicts:
        warnings.append(f"target {task_id}: {conflicts} conflicting duplicate records; retained lowest Vina per SMILES")
    if composite_mismatches:
        warnings.append(f"target {task_id}: {composite_mismatches} rows have composite inconsistent with reported weights")
    return list(records.values()), dict(task_id=task_id, source_files=files,
                                        unique_smiles=len(unique_smiles),
                                        rows_with_metrics=metric_rows,
                                        unique_evaluated=len(records),
                                        duplicate_conflicts=conflicts,
                                        composite_mismatches=composite_mismatches)


def choose(records, selection):
    if not records:
        return None
    if selection == "docking":
        return min(records, key=lambda r: (r["vina"], r["mol_weight"], r["SMILES"]))
    return max(records, key=lambda r: (r["composite"], -r["mol_weight"], r["SMILES"]))


def is_hit(row):
    return (row["vina"] < THRESHOLDS["vina"] and
            row["qed"] > THRESHOLDS["qed"] and
            row["sa"] > THRESHOLDS["sa"])


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe(rows, denominator):
    summary = {"n_selected": len(rows), "n_hit": sum(r["hit"] for r in rows),
               "hit_rate_all_targets": sum(r["hit"] for r in rows) / denominator}
    for field in (*METRICS, "mol_weight", "heavy_atom_count", "ligand_efficiency"):
        vals = [r[field] for r in rows]
        summary[field] = {"mean": mean(vals) if vals else None,
                          "median": median(vals) if vals else None,
                          "sd": stdev(vals) if len(vals) > 1 else None}
    return summary


def main():
    args = parse_args()
    if args.total_targets < 1:
        raise ValueError("--total-targets must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    coverage = []
    selected = {"composite": [], "docking": []}
    missing = []
    for task_id in range(args.total_targets):
        candidates, counts = read_target(args.root, task_id, args, warnings)
        coverage.append(counts)
        if not candidates:
            missing.append(task_id)
            continue
        for selection in selected:
            row = dict(choose(candidates, selection))
            row["hit"] = int(is_hit(row))  # AFTER Top-1 selection, never a selection filter
            selected[selection].append(row)

    for selection, rows in selected.items():
        write_csv(args.output_dir / f"CrossDocked2020_{selection}_top1.csv", EXPORT_FIELDS, rows)
    write_csv(args.output_dir / "CrossDocked2020_target_coverage.csv",
              ("task_id", "source_files", "unique_smiles", "rows_with_metrics",
               "unique_evaluated", "duplicate_conflicts", "composite_mismatches"), coverage)
    report = {"total_targets": args.total_targets, "targets_with_evaluable_molecules":
              args.total_targets - len(missing), "missing_target_ids": missing,
              "candidate_sources": ["<root>/<id>/molecules.csv", "<root>/cache/<id>/molecules.csv",
                                    "<root>/cache/<id>/molecules_pool.csv"],
              "duplicate_policy": "canonical SMILES; lowest Vina for duplicate records (monitor.py)",
              "selection_policy": "unconditional Top-1; hit applied after selection",
              "hit_thresholds_strict": THRESHOLDS,
              "statistics": {selection: describe(rows, args.total_targets) for selection, rows in selected.items()},
              "warnings": warnings}
    (args.output_dir / "CrossDocked2020_evaluation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"total_targets": report["total_targets"],
                      "targets_with_evaluable_molecules": report["targets_with_evaluable_molecules"],
                      "missing_target_ids": missing,
                      "statistics": report["statistics"], "warning_count": len(warnings)}, indent=2))
    if warnings:
        print("[WARNING] See evaluation_summary.json for duplicate/composite discrepancies", file=sys.stderr)
    if missing and not args.allow_incomplete:
        print("[ERROR] Incomplete target coverage; output files are provisional. "
              "Use --allow-incomplete only if explicitly reporting missing targets.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
