"""Run CadreAI against ELILLM CrossDocked2020 structure-files-test pairs.

Required files: {index}-protein.pdb and {index}-ligand.sdf.
Run from the CadreAI project directory, with the updated modules alongside it.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import csv
import random
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

from CadreAI import CadreAI
from DrugExpert import DrugExpert

ERROR_LOCK = threading.Lock()
CSV_COLUMNS = ["SMILES", "vina", "qed", "sa", "composite"]


@dataclass(frozen=True)
class TargetInfo:
    index: str
    pdb_id: str
    chain_id: str
    dataset: str


def parse_arguments():
    parser = argparse.ArgumentParser(description="CadreAI / ELILLM CrossDocked2020")
    parser.add_argument("--commander_model", default="deepseek-flash")
    parser.add_argument("--round", default="2")
    parser.add_argument("--crossdocked_dir", default="datasets/crossdocked/structure-files-test")
    parser.add_argument("--test_info", default="datasets/crossdocked/test-info.csv")
    parser.add_argument("--dataset", default="CrossDocked-test")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--target_ids", nargs="*", default=None)
    return parser.parse_args()


def normalized_index(value: str) -> str:
    value = value.strip()
    assert value.isdecimal(), f"Expected numeric CrossDocked index, got {value!r}"
    return str(int(value))


def read_target_map(csv_path: Path, dataset: str) -> dict[str, TargetInfo]:
    assert csv_path.is_file(), f"Missing test-info.csv: {csv_path}"
    mapping = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "index", "pdb_id", "chain_id"}
        assert reader.fieldnames is not None
        assert required.issubset(reader.fieldnames), f"Missing columns: {required - set(reader.fieldnames)}"
        for row in reader:
            if row["dataset"].strip() != dataset:
                continue
            idx = normalized_index(row["index"])
            pdb_id = row["pdb_id"].strip().lower()
            chain_id = row["chain_id"].strip()
            assert re.fullmatch(r"[a-z0-9]{4}", pdb_id), f"Invalid PDB ID for {idx}: {pdb_id!r}"
            assert chain_id, f"Missing chain_id for {idx}"
            assert idx not in mapping, f"Duplicate dataset/index in {csv_path}: {dataset}/{idx}"
            mapping[idx] = TargetInfo(idx, pdb_id, chain_id, dataset)
    assert mapping, f"No rows for dataset {dataset!r} in {csv_path}"
    return mapping


def discover_targets(structures: Path) -> list[str]:
    assert structures.is_dir(), f"Missing structure directory: {structures}"
    proteins = {p.name[:-len("-protein.pdb")] for p in structures.glob("*-protein.pdb")}
    ligands = {p.name[:-len("-ligand.sdf")] for p in structures.glob("*-ligand.sdf")}
    assert proteins == ligands, f"Unpaired targets: {sorted(proteins ^ ligands)[:20]}"
    assert proteins, f"No protein/ligand pairs found in {structures}"
    by_index = {}
    for raw in proteins:
        idx = normalized_index(raw)
        assert idx not in by_index, f"Duplicate numeric index from filenames: {raw} and {by_index.get(idx)}"
        by_index[idx] = raw
        assert (structures / f"{raw}-protein.pdb").stat().st_size > 0
        assert (structures / f"{raw}-ligand.sdf").stat().st_size > 0
    return [by_index[idx] for idx in sorted(by_index, key=int)]


def read_protein(pdb_file: Path, expected_chain: str) -> tuple[str, str]:
    """Read actual receptor coordinates; prefer the intended PDB chain.

    Some ELILLM processed receptor PDBs renumber a single chain to 'A'.
    In that special case the sole chain is accepted and logged explicitly.
    """
    structure = PDBParser(QUIET=True).get_structure(pdb_file.stem, str(pdb_file))
    model = next(structure.get_models())
    sequences = {}
    for chain in model:
        sequence = "".join(
            seq1(residue.resname, custom_map={"UNK": "X"})
            for residue in chain
            if residue.id[0] == " " and "CA" in residue
        )
        if sequence:
            assert sequence.isalpha(), f"Invalid sequence for chain {chain.id} in {pdb_file}"
            sequences[chain.id] = sequence
    assert sequences, f"No amino-acid chain detected in {pdb_file}"
    if expected_chain in sequences:
        actual_chain = expected_chain
    elif len(sequences) == 1:
        actual_chain = next(iter(sequences))
        print(f"[CHAIN-REMAP] {pdb_file.name}: CSV chain={expected_chain}, receptor chain={actual_chain}", flush=True)
    else:
        raise ValueError(
            f"Expected chain {expected_chain!r} not found in {pdb_file}; "
            f"available chains={list(sequences)}. Refusing to select an arbitrary chain."
        )
    return sequences[actual_chain], actual_chain


def complete(molecules_path: Path, smiles_path: Path) -> bool:
    if not (molecules_path.is_file() and smiles_path.is_file()):
        return False
    if not smiles_path.read_text(encoding="utf-8").strip():
        return False
    with molecules_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames == CSV_COLUMNS and any(row.get("SMILES", "").strip() for row in reader)


def run_target(raw_idx, info, args, structures, output_dir, cache_dir, drug_expert, error_log):
    receptor = structures / f"{raw_idx}-protein.pdb"
    reference = structures / f"{raw_idx}-ligand.sdf"
    target_dir = output_dir / raw_idx
    molecules_file = target_dir / "molecules.csv"
    smiles_file = target_dir / "final_molecule.smiles"
    if complete(molecules_file, smiles_file):
        print(f"[SKIP] {raw_idx}: completed", flush=True)
        return "skipped"

    locks_dir = output_dir / "locks"
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f"{raw_idx}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[LOCKED] {raw_idx}: another process owns this target", flush=True)
        return "locked"
    os.close(lock_fd)

    try:
        if complete(molecules_file, smiles_file):
            return "skipped"
        target_dir.mkdir(parents=True, exist_ok=True)
        sequence, actual_chain = read_protein(receptor, info.chain_id)
        protein_title = f"PDB {info.pdb_id}, chain {info.chain_id}"
        print(
            f"[START] index={raw_idx} pdb_id={info.pdb_id} "
            f"csv_chain={info.chain_id} receptor_chain={actual_chain} "
            f"length={len(sequence)} receptor={receptor.name} reference={reference.name}",
            flush=True,
        )

        # Keep the original per-index cache/output layout for resumption.
        target_cache = cache_dir / raw_idx
        target_cache.mkdir(parents=True, exist_ok=True)
        agent = CadreAI(
            commander_model=args.commander_model,
            drug_model_itself=drug_expert,
            max_iterations=args.max_iterations,
            cache_dir=str(target_cache) + os.sep,
            description=raw_idx,
            crossdocked_dir=str(structures),
            target_map_file=None,
        )
        result = agent.run_drug_discovery_cycle(
            protein_sequence=sequence,
            protein_title=protein_title,
            pdb_id=info.pdb_id,     # ProteinExpert/cache: actual PDB identity
            target_index=raw_idx,   # BindingSimulator: local numeric ELILLM files
            # protein_lookup_id=info.pdb_id,
        )
        final_smiles = result["final_smiles"]
        molecules = result["molecule_list"]
        assert isinstance(final_smiles, str) and final_smiles.strip()
        assert molecules, f"No molecules produced for {raw_idx}"
        for molecule in molecules:
            assert len(molecule) == len(CSV_COLUMNS), f"Invalid molecule row for {raw_idx}: {molecule!r}"

        tmp_csv = target_dir / "molecules.csv.tmp"
        tmp_smiles = target_dir / "final_molecule.smiles.tmp"
        with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(molecules)
        tmp_smiles.write_text(final_smiles + "\n", encoding="utf-8")
        os.replace(tmp_csv, molecules_file)
        os.replace(tmp_smiles, smiles_file)
        print(f"[DONE] {raw_idx} ({info.pdb_id}): {len(molecules)} molecules", flush=True)
        return "completed"
    except Exception as exc:
        details = traceback.format_exc()
        print(f"[ERROR] {raw_idx} ({info.pdb_id}): {type(exc).__name__}: {exc}", flush=True)
        with ERROR_LOCK:
            with error_log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{raw_idx}] pdb_id={info.pdb_id}\n{details}\n")
        return "failed"
    finally:
        lock_path.unlink()


def main():
    args = parse_arguments()
    assert os.environ.get("DEEPSEEK_KEY"), "Set DEEPSEEK_KEY in the environment"
    assert args.workers > 0
    assert args.max_iterations >= 0
    assert args.max_targets is None or args.max_targets > 0

    structures = Path(args.crossdocked_dir).resolve()
    mapping = read_target_map(Path(args.test_info).resolve(), args.dataset)
    target_ids = discover_targets(structures)
    missing = [idx for idx in target_ids if normalized_index(idx) not in mapping]
    assert not missing, f"Missing {args.dataset} mapping for target IDs: {missing[:20]}"
    if args.target_ids is not None:
        requested = {normalized_index(idx) for idx in args.target_ids}
        available = {normalized_index(idx) for idx in target_ids}
        assert requested <= available, f"Unknown targets: {sorted(requested - available)}"
        target_ids = [idx for idx in target_ids if normalized_index(idx) in requested]
    random.Random(args.seed).shuffle(target_ids)
    if args.max_targets is not None:
        target_ids = target_ids[:args.max_targets]
    assert target_ids, "No targets selected"

    model_tag = args.commander_model.replace(":", "_").replace("-", "_")
    # output_dir = Path(args.output_dir or f"revised_vina_output_CD_{model_tag}_MT_with_insight_opt_round{args.round}").resolve()
    output_dir = Path(args.output_dir or f"CadreAI Pro").resolve()
    # output_dir = Path(args.output_dir or f"Focus_vina_only_output_CD_{model_tag}_MT_with_insight_opt_round{args.round}").resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "error_log.txt"
    print(f"[DATA] {structures}\n[MAP] {Path(args.test_info).resolve()}\n"
          f"[TARGETS] {len(target_ids)}\n[OUTPUT] {output_dir}\n[WORKERS] {args.workers}", flush=True)

    drug_expert = DrugExpert(
        model_name_or_id="OpenDFM/ChemDFM-v1.5-8B",
        cache_file="drug_cache.json",
        need_cache=True,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        cache_dir=str(cache_dir),
    )
    totals = {"completed": 0, "skipped": 0, "locked": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(run_target, idx, mapping[normalized_index(idx)], args,
                        structures, output_dir, cache_dir, drug_expert, error_log)
            for idx in target_ids
        ]
        for future in as_completed(futures):
            totals[future.result()] += 1
            print(f"[PROGRESS] {sum(totals.values())}/{len(target_ids)} {totals}", flush=True)
    print(f"[FINISHED] {totals}", flush=True)


if __name__ == "__main__":
    main()
