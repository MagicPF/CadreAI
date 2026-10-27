"""DrugExpert with 30 primary trials and 20 MW-relaxed fallback trials.

Requires the project's original utils.py (calcScore, save_to_llama_format,
filter_chinese_characters), PyTorch, Transformers, RDKit and Meeko.
SA from calcScore MUST be normalized to [0, 1], with larger values better.
After all trials fail, optionally return the last preparable property-rejected
candidate, relaxing MW/QED/SA thresholds but never structural validity.
"""
import json
import logging
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation

try:
    from meeko import PDBQTWriterLegacy
except ImportError:
    PDBQTWriterLegacy = None

from utils import calcScore, save_to_llama_format, filter_chinese_characters

logger = logging.getLogger(__name__)


class CandidateRejected(ValueError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


class DrugExpert:
    ALLOWED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35})

    def __init__(
        self,
        model_name_or_id="OpenDFM/ChemDFM-v1.5-8B",
        device="cuda",
        cache_file="drug_cache.json",
        data_file="drug_expert_data.json",
        need_cache=False,
        cache_dir="./jsons/",
        primary_attempts=20,
        fallback_attempts=20,
        primary_mw_max=600.0,
        # primary_mw_max=400.0,
        fallback_mw_max=800.0,
        qed_min=0.20,
        normalized_sa_min=0.40,
        trial_log_file="generation_trials.jsonl",
        embedding_seed=None,
        return_last_valid_on_exhaustion=True,
    ):
        for name, value in (("primary_attempts", primary_attempts),
                            ("fallback_attempts", fallback_attempts)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if primary_attempts < 1:
            raise ValueError("primary_attempts must be at least 1")
        numbers = (primary_mw_max, fallback_mw_max, qed_min, normalized_sa_min)
        if not all(math.isfinite(float(x)) for x in numbers):
            raise ValueError("All property limits must be finite")
        if not 0 < primary_mw_max <= fallback_mw_max:
            raise ValueError("Require 0 < primary_mw_max <= fallback_mw_max")
        if not 0 <= qed_min <= 1 or not 0 <= normalized_sa_min <= 1:
            raise ValueError("QED and normalized SA limits must be in [0, 1]")

        self.device = device
        self.need_cache = need_cache
        self.model_name_or_id = model_name_or_id
        self.primary_attempts = primary_attempts
        self.fallback_attempts = fallback_attempts
        self.primary_mw_max = float(primary_mw_max)
        self.fallback_mw_max = float(fallback_mw_max)
        self.qed_min = float(qed_min)
        self.normalized_sa_min = float(normalized_sa_min)
        self.embedding_seed = embedding_seed
        self.return_last_valid_on_exhaustion = bool(return_last_valid_on_exhaustion)
        self.last_generation_info = None

        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.cache_file = str(directory / cache_file)
        self.data_file = str(directory / data_file)
        self.trial_log_file = str(directory / trial_log_file) if trial_log_file else None
        if not Path(self.data_file).exists():
            self._write_json(self.data_file, [])
        self.cache = self._load_cache() if need_cache else {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_id, torch_dtype=torch.float16, device_map=device
        )

    @staticmethod
    def _write_json(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_cache(self):
        path = Path(self.cache_file)
        if path.exists():
            with path.open(encoding="utf-8-sig") as handle:
                return json.load(handle)
        return {}

    def _save_cache(self):
        if self.need_cache:
            self._write_json(self.cache_file, self.cache)

    def _append_trial(self, record):
        # Use a separate cache_dir per target/process, as in the original runner.
        # This class does not coordinate shared multi-process JSON writers.
        if self.trial_log_file:
            path = Path(self.trial_log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _validate_candidate(self, text, mw_cap, enforce_property_limits=True, mw_min=0.0):
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            raise CandidateRejected("invalid_smiles", "Invalid or empty SMILES")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise CandidateRejected("multiple_fragments", "Multiple fragments detected")
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in self.ALLOWED_ELEMENTS:
                raise CandidateRejected("forbidden_element", f"Forbidden element: {atom.GetSymbol()}")
        try:
            _, qed, _, mw, sa_score, _ = calcScore(mol)
            qed, mw, sa_score = float(qed), float(mw), float(sa_score)
        except Exception as exc:
            raise CandidateRejected("property_calculation", f"calcScore failed: {exc}") from exc
        if not all(math.isfinite(v) for v in (qed, mw, sa_score)):
            raise CandidateRejected("nonfinite_property", "MW, QED or SA is non-finite")
        if mw <= 0 or not 0 <= qed <= 1 or not 0 <= sa_score <= 1:
            raise CandidateRejected("property_scale", "Require MW>0 and QED/normalized SA in [0,1]")
        # QED/SA are NOT relaxed in fallback; check them before the MW limit.
        if enforce_property_limits and qed < self.qed_min:
            raise CandidateRejected("qed", f"QED {qed:.4f} below {self.qed_min:.4f}")
        if enforce_property_limits and sa_score < self.normalized_sa_min:
            raise CandidateRejected("sa", f"Normalized SA {sa_score:.4f} below {self.normalized_sa_min:.4f}")
        if enforce_property_limits and mw > mw_cap:
            raise CandidateRejected("mw", f"MW {mw:.4f} exceeds {mw_cap:.1f} Da")
        if enforce_property_limits and mw < mw_min:
            raise CandidateRejected("mw_min", f"MW {mw:.4f} below active exploration range {mw_min:.1f} Da")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        properties = dict(mw=mw, qed=qed, normalized_sa=sa_score,
                          heavy_atoms=mol.GetNumHeavyAtoms())
        return mol, canonical, properties

    def _prepare_for_docking(self, mol):
        hydrogenated = Chem.AddHs(mol)
        try:
            if self.embedding_seed is None:
                status = AllChem.EmbedMolecule(hydrogenated)
            else:
                status = AllChem.EmbedMolecule(hydrogenated, randomSeed=int(self.embedding_seed))
        except Exception as exc:
            raise CandidateRejected("embedding", f"3D embedding failed: {exc}") from exc
        if status == -1:
            raise CandidateRejected("embedding", "3D conformation generation failed")
        try:
            preparator = MoleculePreparation()
            setups = preparator.prepare(hydrogenated)
            if not setups:
                raise ValueError("Meeko preparation returned no setup")
            if PDBQTWriterLegacy is not None and isinstance(setups, (list, tuple)):
                result = PDBQTWriterLegacy.write_string(setups[0])
            else:
                # Compatibility with the legacy API used in the original DE.
                result = preparator.write_pdbqt_string()
            if isinstance(result, tuple):
                pdbqt, success, error = result
                if not success:
                    raise ValueError(str(error))
            else:
                pdbqt = result
        except Exception as exc:
            raise CandidateRejected("meeko", f"Meeko preparation/write failed: {exc}") from exc
        if not isinstance(pdbqt, str) or not pdbqt.strip():
            raise CandidateRejected("pdbqt", "Empty or invalid PDBQT writer output")
        lines = [line.strip() for line in pdbqt.splitlines()]
        # Rigid ligands may have TORSDOF 0 and no BRANCH: this is valid.
        if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
            raise CandidateRejected("pdbqt", "PDBQT contains no atom records")
        if "ROOT" not in lines or "ENDROOT" not in lines:
            raise CandidateRejected("pdbqt", "Missing PDBQT ROOT/ENDROOT")
        if not any(line.startswith("TORSDOF ") for line in lines):
            raise CandidateRejected("pdbqt", "Missing PDBQT TORSDOF")
        return pdbqt

    def generate_molecule(self, properties_text, do_sample=True, top_k=50,
                          top_p=0.8, temperature=1.0, max_new_tokens=128,
                          scaffold=False, generation_mode="standard",
                          large_mw_min=600.0, large_mw_max=800.0,
                        #   large_mw_min=500.0, large_mw_max=800.0,
                          strategy_context="", mode=None) -> str:
        """Try 30 primary + 20 fallback trials, then return a preparable candidate.

        scaffold remains API-compatible but does not change the property limits.
        No docking is performed here, so Vina and LE are not evaluated or invented.
        Exhaustion return relaxes MW/QED/SA only, is explicitly logged, and can
        be disabled with return_last_valid_on_exhaustion=False.
        """
        if mode is not None:
            if generation_mode != "standard" and generation_mode != mode:
                raise ValueError("Conflicting mode and generation_mode")
            generation_mode = mode
        run_id = uuid4().hex
        if generation_mode not in {"standard", "direct_large"}:
            raise ValueError("generation_mode must be standard or direct_large")
        if generation_mode == "direct_large":
            if not all(math.isfinite(float(v)) for v in (large_mw_min,large_mw_max)):
                raise ValueError("Large-molecule size limits must be finite")
            if not 0 < large_mw_min < large_mw_max <= self.fallback_mw_max:
                raise ValueError("Require 0 < large_mw_min < large_mw_max <= fallback_mw_max")
        failures = Counter()
        total = self.primary_attempts + self.fallback_attempts
        self.last_generation_info = None
        emergency_candidates = []
        generation_config = GenerationConfig(
            do_sample=do_sample, top_k=top_k, top_p=top_p,
            temperature=temperature, max_new_tokens=max_new_tokens,
            repetition_penalty=1.02, eos_token_id=self.tokenizer.eos_token_id,
        )
        for attempt_index in range(total):
            fallback = attempt_index >= self.primary_attempts
            stage = "fallback" if fallback else "primary"
            stage_attempt = attempt_index - self.primary_attempts + 1 if fallback else attempt_index + 1
            mw_cap = self.fallback_mw_max if fallback else self.primary_mw_max
            mw_min = 0.0
            if generation_mode == "direct_large":
                stage,stage_attempt="direct_large",attempt_index+1
                mw_min,mw_cap=float(large_mw_min),float(large_mw_max)
            if generation_mode == "standard" and fallback and stage_attempt == 1:
                logger.warning("Primary stage exhausted after %d trials; fallback MW cap %.1f Da. Failures: %s",
                               self.primary_attempts, mw_cap, dict(failures))
            if generation_mode == "direct_large":
                size_guidance = (
                    f"Explore a compact larger ligand in the {mw_min:g}-{mw_cap:g} Da range. "
                    "Use target-justified growth, ring extension or pharmacophore placement; "
                    "do not inflate mass using redundant alkyl chains or heavy atoms. "
                )
            else:
                size_guidance = "Prefer compact molecules around 300-550 Da. "
            input_text = (
                "[Round 0]\nHuman: Generate a molecule meeting these requirements: "
                f"{properties_text}\n"
                f"{size_guidance}Prefer QED >= 0.5 "
                "and normalized synthetic accessibility >= 0.6. "
                "These are preferences; preserve target-relevant structural features. "
                "Avoid unnecessary bulky substituents and long flexible chains. "
                f"The current molecular-weight ceiling is {mw_cap:g} Da; "
                "do not increase molecular size merely to approach that ceiling. "
                "Return only one valid, single-component SMILES, without explanation.\n"
                f"Stage-specific design guidance: {str(strategy_context)[:1500]}\n"
                "Assistant: The generated SMILES is:"
            )
            record = dict(event="generation_trial", run_id=run_id, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          attempt=attempt_index+1, stage=stage, stage_attempt=stage_attempt,
                          mw_cap=mw_cap, qed_min=self.qed_min,
                          normalized_sa_min=self.normalized_sa_min, scaffold=bool(scaffold),
                          generation_mode=generation_mode, mw_min=mw_min,
                          do_sample=do_sample, top_k=top_k, top_p=top_p,
                          temperature=temperature, max_new_tokens=max_new_tokens)
            try:
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, generation_config=generation_config)
                tokens = outputs[:, inputs.input_ids.shape[-1]:]
                generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            except Exception as exc:
                # Infrastructure/model failures should surface, not consume 50
                # pointless property-filter retries (e.g. CUDA out of memory).
                record.update(accepted=False, reason="generation_error", detail=str(exc))
                self._append_trial(record)
                raise
            record["generated_text"] = generated_text
            try:
                mol, smiles, properties = self._validate_candidate(generated_text, mw_cap, mw_min=mw_min)
                record.update(properties)
                self._prepare_for_docking(mol)
            except CandidateRejected as exc:
                failures[exc.reason] += 1
                record.update(accepted=False, reason=exc.reason, detail=str(exc))
                if self.return_last_valid_on_exhaustion and exc.reason in {"mw", "mw_min", "qed", "sa"}:
                    # Retain text and provenance only. Do expensive 3D/Meeko
                    # preparation only if all ordinary trials are exhausted.
                    emergency_candidates.append((generated_text, dict(record)))
                self._append_trial(record)
                logger.warning("DE trial %d/%d [%s %d]: %s", attempt_index+1,total,stage,stage_attempt,exc)
                continue
            record.update(accepted=True, smiles=smiles, reason="accepted", failures_before_success=dict(failures))
            self._append_trial(record)
            self.last_generation_info = record
            save_to_llama_format("Generate a molecule with specific properties.",
                                 properties_text, smiles, self.data_file)
            logger.info("DE accepted trial %d [%s]: MW=%.2f QED=%.3f SA=%.3f",
                        attempt_index+1,stage,properties['mw'],properties['qed'],properties['normalized_sa'])
            return smiles
        if self.return_last_valid_on_exhaustion:
            seen = set()
            for text, source_record in reversed(emergency_candidates):
                try:
                    mol, smiles, properties = self._validate_candidate(
                        text, mw_cap=self.fallback_mw_max, enforce_property_limits=False
                    )
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    self._prepare_for_docking(mol)
                except CandidateRejected as exc:
                    self._append_trial(dict(
                        event="exhaustion_preparation_rejected", run_id=run_id,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        candidate_attempt=source_record['attempt'],
                        accepted=False, reason=exc.reason, detail=str(exc),
                    ))
                    continue
                info = dict(source_record)
                info.update(properties)
                info.update(
                    event="forced_return", accepted=True, stage="forced_return",
                    source_stage=source_record['stage'],
                    candidate_attempt=source_record['attempt'], attempt=total,
                    generation_attempts=total, smiles=smiles,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    reason="all_trials_exhausted_return_last_preparable",
                    original_rejection=source_record['reason'],
                    property_limits_relaxed=True, failures_before_success=dict(failures),
                )
                self._append_trial(info)
                self.last_generation_info = info
                save_to_llama_format("Generate a molecule with specific properties.",
                                     properties_text, smiles, self.data_file)
                logger.warning("DE forced return after %d trials: candidate from trial %d, "
                               "MW=%.2f QED=%.3f SA=%.3f; property limits relaxed",
                               total,source_record['attempt'],properties['mw'],
                               properties['qed'],properties['normalized_sa'])
                return smiles
        self.last_generation_info = dict(run_id=run_id, accepted=False, attempts=total,
                                         primary_attempts=self.primary_attempts,
                                         fallback_attempts=self.fallback_attempts, failures=dict(failures))
        raise ValueError(f"Failed after {self.primary_attempts} primary trials (MW<={self.primary_mw_max:g}) "
                         f"and {self.fallback_attempts} fallback trials (MW<={self.fallback_mw_max:g}); "
                         f"rejection counts={dict(failures)}")

    def predict_toxicity_and_promiscuity(self, smiles: str, max_new_tokens=128) -> dict:
        instruction = "Analyze the toxicity and promiscuity of a given molecule."
        report = {}
        for round_num, prompt in enumerate([
            "Analyze the molecular structure and identify functional groups associated with toxicity.",
            "Predict how this molecule is likely to be metabolized in the human body.",
            "Check if this molecule shares structural similarity to known toxic compounds.",
        ], start=1):
            input_text = f"[Round {round_num}]\nHuman: {prompt}\nSMILES: {smiles}\nAssistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True)[0]
            report[prompt] = text
            save_to_llama_format(instruction,f"SMILES: {smiles}, Task: {prompt}",text,self.data_file)
        return report

    def chat(self, text_input: str, instruction="Engage in the inference about drug discovery and molecular design.",
             max_new_tokens=512) -> str:
        input_text = f"Human: {text_input} Assistant: "
        config = GenerationConfig(do_sample=True,top_k=50,top_p=0.9,temperature=1.0,
                                  max_new_tokens=max_new_tokens,repetition_penalty=1.2,
                                  eos_token_id=self.tokenizer.eos_token_id)
        inputs = self.tokenizer(instruction+" "+input_text,return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs,generation_config=config)
        response = self.tokenizer.batch_decode(outputs[:,inputs.input_ids.shape[-1]:],skip_special_tokens=True)[0]
        save_to_llama_format(instruction,text_input,response,self.data_file)
        return filter_chinese_characters(response.strip())

    def compute_properties(self, smiles):
        if not isinstance(smiles,str) or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            logp,qed,tpsa,mw,sa_score,np_score = calcScore(mol)
        except Exception as exc:
            logger.warning("calcScore failed: %s",exc)
            return None
        props = {
            "OctanolWaterPartitionCoefficientS(↑)":logp,
            "QuantitativeEstimateOfDruglikeness(↑)":qed,
            "TopologicalPolarSurfaceArea":tpsa,
            "MolecularWeight":mw,
            "NormalizedSyntheticAccessibilityScore(↑)":sa_score,
            "NaturalProductLikenessScore(↑)":np_score,
        }
        try:
            Chem.Kekulize(mol)
            props['MolMR'] = Descriptors.MolMR(mol)
        except Exception as exc:
            logger.warning("Kekulization failed: %s",exc)
            props['MolMR'] = None
        return props
