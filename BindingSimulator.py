"""ELILLM CrossDocked2020 smina simulator.

Uses the EXACT benchmark receptor/reference pair from
  datasets/crossdocked/structure-files-test/{index}-protein.pdb
  datasets/crossdocked/structure-files-test/{index}-ligand.sdf

The docking and ligand-preparation parameters follow ELILLM/tools/docking.py.
Error handling and subprocess execution intentionally use safer semantics.
"""

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from typing import Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

SMINA_BINARY = "smina"
CROSSDOCKED_DIR = Path("datasets/crossdocked/structure-files-test")
# SMINA_AUTOBOX_ADD = 6
SMINA_AUTOBOX_ADD = 6
SMINA_SEED = 1234
SMINA_EXHAUSTIVENESS = 9
SMINA_CPU = 4
SMINA_TIMEOUT = 1800
EMBED_SEED = 1
FAILED_AFFINITY = 500.0
LOGGER = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]
_SCORE_ROW = re.compile(
    r"\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
)


def _mode_one_score(stdout: str) -> float:
    scores = []
    for line in stdout.splitlines():
        match = _SCORE_ROW.fullmatch(line)
        if match and int(match.group(1)) == 1:
            scores.append(float(match.group(2)))
    if len(scores) != 1:
        raise ValueError(f"Expected one smina mode-1 affinity, got {len(scores)}.\n{stdout[-3000:]}")
    return scores[0]


def _prepare_ligand(smiles: str, path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles!r}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    status = AllChem.EmbedMolecule(mol, params)
    assert status == 0, f"ETKDGv3 failed for SMILES: {smiles!r}"
    # Match ELILLM: default MMFF options; no custom convergence threshold.
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    assert path.is_file() and path.stat().st_size > 0


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class BindingSimulator:
    def __init__(
        self,
        out_dir: PathLike = "boltz_output",
        use_msa_server: bool = True,
        accelerator: str = "gpu",
        cache_file: PathLike = "binding_cache.json",
        need_cache: bool = False,
        crossdocked_dir: PathLike = CROSSDOCKED_DIR,
        smina_binary: str = SMINA_BINARY,
        target_map_file: Optional[PathLike] = None,
    ):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_msa_server = use_msa_server  # API compatibility only.
        self.accelerator = accelerator        # ELILLM smina executes on CPU.
        self.cache_file = Path(cache_file).resolve()
        self.need_cache = need_cache
        self.crossdocked_dir = Path(crossdocked_dir).resolve()
        self.smina_binary = smina_binary
        self._cache_lock = threading.Lock()
        self.cache = self._load_cache() if need_cache else {}
        self.target_map = {}
        if target_map_file is not None:
            mapping = json.loads(Path(target_map_file).read_text(encoding="utf-8"))
            assert isinstance(mapping, dict), "Target mapping must be a JSON object"
            self.target_map = {str(k): str(v) for k, v in mapping.items()}

    def _load_cache(self) -> dict:
        if not self.cache_file.exists():
            return {}
        value = json.loads(self.cache_file.read_text(encoding="utf-8"))
        assert isinstance(value, dict), "Cache must be a JSON object"
        return value

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".binding-cache-", suffix=".json", dir=self.cache_file.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.cache, stream, indent=2)
            os.replace(tmp, self.cache_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def list_targets(self) -> list[str]:
        """List only targets with both exact ELILLM reference files."""
        assert self.crossdocked_dir.is_dir(), f"Missing test structure directory: {self.crossdocked_dir}"
        proteins = {p.name.removesuffix("-protein.pdb") for p in self.crossdocked_dir.glob("*-protein.pdb")}
        ligands = {p.name.removesuffix("-ligand.sdf") for p in self.crossdocked_dir.glob("*-ligand.sdf")}
        return sorted(proteins & ligands, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))

    def _resolve_inputs(
        self,
        pdb_id: str,
        target_index: Optional[Union[int, str]],
        receptor_path: Optional[PathLike],
        reference_ligand_path: Optional[PathLike],
    ) -> tuple[Path, Path]:
        if (receptor_path is None) != (reference_ligand_path is None):
            raise ValueError("Provide both receptor_path and reference_ligand_path, or neither")
        if receptor_path is not None:
            receptor = Path(receptor_path).resolve()
            reference = Path(reference_ligand_path).resolve()
        else:
            if target_index is not None:
                target = str(target_index)
            elif str(pdb_id) in self.target_map:
                target = self.target_map[str(pdb_id)]
            elif str(pdb_id).isdigit():
                target = str(pdb_id)
            else:
                raise ValueError(
                    f"Cannot map DrugVoyager target {pdb_id!r} to an ELILLM test index. "
                    "Pass target_index=..., set target_map_file, or provide both exact structure paths. "
                    "PDB IDs and ELILLM numeric indices are not interchangeable."
                )
            if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
                raise ValueError(f"Unsafe test-set target identifier: {target!r}")
            receptor = self.crossdocked_dir / f"{target}-protein.pdb"
            reference = self.crossdocked_dir / f"{target}-ligand.sdf"
        if not receptor.is_file():
            raise FileNotFoundError(f"Receptor missing: {receptor}")
        if not reference.is_file():
            raise FileNotFoundError(f"Reference ligand missing: {reference}")
        if receptor.stat().st_size == 0 or reference.stat().st_size == 0:
            raise ValueError("Receptor/reference ligand file is empty")
        return receptor.resolve(), reference.resolve()

    def CaculateAffinity(
        self,
        protein_seq: str,
        drug_smiles: str,
        output_format: str = "pdb",
        box_size: float = 20,
        exhaustiveness: int = SMINA_EXHAUSTIVENESS,
        pdb_id: str = "NotAvailable",
        reference_ligand_path: Optional[PathLike] = None,
        receptor_path: Optional[PathLike] = None,
        chain: Optional[str] = None,
        autobox_add: float = SMINA_AUTOBOX_ADD,
        seed: int = SMINA_SEED,
        cpu: int = SMINA_CPU,
        target_index: Optional[Union[int, str]] = None,
        num_smina: int = 1,
        save_pose: bool = False,
    ) -> dict:
        """Evaluate SMILES with the ELILLM CrossDocked reference box.

        Returns {'binding_affinity': float}. Legacy unused arguments are accepted.
        Any non-default docking parameters deviate from the reference protocol.
        """
        try:
            del protein_seq, output_format, box_size, chain
            assert isinstance(drug_smiles, str) and drug_smiles.strip(), "Empty SMILES"
            assert int(num_smina) >= 1
            # assert int(cpu) >= 1 and int(exhaustiveness) >= 1 and float(autobox_add) >= 0
            executable = shutil.which(self.smina_binary)
            if executable is None:
                raise FileNotFoundError(f"smina not found: {self.smina_binary}")
            receptor, reference = self._resolve_inputs(pdb_id, target_index, receptor_path, reference_ligand_path)
            payload = {
                "protocol": "ELILLM-crossdocked-2020-v2", "smiles": drug_smiles,
                "receptor": _identity(receptor), "reference": _identity(reference),
                "autobox_add": float(autobox_add), "seed": int(seed),
                "exhaustiveness": int(exhaustiveness), #"cpu": int(cpu),
                "embed_seed": EMBED_SEED, "num_smina": int(num_smina), "executable": executable,
            }
            cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            with self._cache_lock:
                if self.need_cache and cache_key in self.cache:
                    return dict(self.cache[cache_key])
            scores = []
            for attempt in range(int(num_smina)):
                with tempfile.TemporaryDirectory(prefix="elillm_smina_", dir=self.out_dir) as tmp:
                    ligand_path = Path(tmp) / "generated_ligand.sdf"
                    _prepare_ligand(drug_smiles, ligand_path)
                    command = [
                        executable, "-r", str(receptor), "-l", str(ligand_path),
                        "--autobox_ligand", str(reference),
                        "--autobox_add", str(autobox_add),
                        "--seed", str(int(seed)),
                        "--exhaustiveness", str(int(exhaustiveness)),
                        # "--cpu", str(int(cpu)),
                    ]
                    if save_pose:
                        pose_path = self.out_dir / f"pose_{cache_key[:16]}_{attempt}.pdb"
                        command.extend(["-o", str(pose_path)])
                    completed = subprocess.run(command, text=True, capture_output=True, timeout=SMINA_TIMEOUT)
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"smina failed (code {completed.returncode})\n"
                            f"Command: {' '.join(command)}\n"
                            f"stdout: {completed.stdout[-3000:]}\nstderr: {completed.stderr[-3000:]}"
                        )
                    scores.append(_mode_one_score(completed.stdout))
            affinity = float(np.asarray(scores, dtype=np.float32).mean())
            if not math.isfinite(affinity):
                raise ValueError(f"Non-finite docking affinity: {affinity}")
            output = {"binding_affinity": affinity}
            print("****************************************************")
            print(f"Current Molecule Affinity:{affinity}")
            print("****************************************************")
            if self.need_cache:
                with self._cache_lock:
                    self.cache[cache_key] = output
                    self._save_cache()
            return output
        except (ValueError, RuntimeError, OSError, AssertionError,
                subprocess.SubprocessError, IndexError, TypeError) as exc:
            LOGGER.warning(
                "[Smina Failed] pdb_id=%s target_index=%s: %s: %s; affinity=%.1f",
                pdb_id, target_index, type(exc).__name__, str(exc)[-1200:], FAILED_AFFINITY,
            )
            return {"binding_affinity": FAILED_AFFINITY, "error": str(exc)[-1200:]}

    CalculateAffinity = CaculateAffinity


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ELILLM references or evaluate one SMILES")
    parser.add_argument("--crossdocked-dir", default=str(CROSSDOCKED_DIR))
    parser.add_argument("--target-index", help="ELILLM test index, e.g. 0")
    parser.add_argument("--smiles", help="Run one smina docking calculation (otherwise only verify dataset)")
    parser.add_argument("--out-dir", default="boltz_output")
    parser.add_argument("--smina-binary", default=SMINA_BINARY)
    args = parser.parse_args()
    simulator = BindingSimulator(out_dir=args.out_dir, crossdocked_dir=args.crossdocked_dir,
                                 smina_binary=args.smina_binary)
    targets = simulator.list_targets()
    print(f"Dataset directory: {simulator.crossdocked_dir}")
    print(f"Complete receptor/reference pairs: {len(targets)}")
    print(f"First target IDs: {targets[:12]}")
    assert targets, "No matching *-protein.pdb / *-ligand.sdf pairs found"
    if args.smiles is not None:
        assert args.target_index is not None, "Specify --target-index for docking"
        result = simulator.CaculateAffinity("", args.smiles, target_index=args.target_index)
        print(f"Affinity: {result['binding_affinity']:.4f} kcal/mol")


if __name__ == "__main__":
    main()
