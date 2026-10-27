# will use full version deepseek, use index book to orgnize the skill libirary, and RAG for inference
from http.client import responses
# import wandb
from ProteinExpert import ProteinExpert
from DrugExpert import DrugExpert
from BindingSimulator import BindingSimulator
from Commander import DeepSeekOllama, Deepseek_official
import sys
from utils import *
import warnings
import os
import json
from difflib import SequenceMatcher, get_close_matches
import random
import math
import csv
import uuid
from rdkit import Chem
from rdkit.Chem import Descriptors
class ConsecutivePoorVinaStop(Exception):
    """Stop the current protein task after repeated poor docking scores."""
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
class CadreAI:
    def __init__(
        self,
        api_key="EMPTY",
        protein_server="http://search-protrek.com/",
        drug_model="OpenDFM/ChemDFM-v1.5-8B",
        drug_model_itself=None,
        commander_model="deepseek-r1:1.5b",
        use_msa_server=True,
        accelerator="gpu",
        device="cuda",
        max_iterations=5,
        skill_lib_file="skill_lib.json",
        history_log_file="history_logs.json",
        cache_dir="./jsons/",
        description="",
        need_save_each_step=False,
        crossdocked_dir="datasets/crossdocked/structure-files-test",
        target_map_file=None,
        poor_vina_threshold=-6.0,
        max_consecutive_poor_vina=10,
    ):
        # wandb.init(
        #     project="CadreAI", name="drug_discovery_with_deepseek_commander",
        #     config={"device": device, "max_iterations": max_iterations}
        # )
        self.protein_expert = ProteinExpert(protein_server, cache_dir="./protein_DB/")
        if drug_model_itself is None:
            self.drug_expert = DrugExpert(
                model_name_or_id=drug_model,
                cache_file="drug_cache.json",
                need_cache=True,
                device=device,
                cache_dir=cache_dir
            )
        else:
            self.drug_expert = drug_model_itself
        self.binding_simulator = BindingSimulator(
            out_dir=os.path.join(cache_dir, f"receptor{description}"),
            use_msa_server=use_msa_server,
            accelerator=accelerator,
            crossdocked_dir=crossdocked_dir,
            target_map_file=target_map_file,
            need_cache=False,
        )
        # 使用官方 DeepSeek 实例
        self.commander = Deepseek_official(key=api_key, cache_dir=os.path.join(cache_dir, "jsons/"))
        self.max_iterations = max_iterations
        self.generation_mode = "standard"
        self.mode_reason = "Default compact-first policy"
        self.mode_history = []
        self._mode_iteration = 0
        self.skill_lib_file = cache_dir + skill_lib_file
        self.history_log_file = cache_dir + history_log_file
        # 技能库改为标题->洞见列表的字典
        self.skill_lib = self._load_skill_lib()
        self.history_logs = self._load_history_logs()
        self.shift_nochange_round = 5
        self.molecule_list = []
        self._elite_pool = []
        self._archive_path = None
        self._improvement_csv_path = os.path.join(cache_dir, "molecules.csv")
        self._saved_seen_smiles = set()
        self._scored_molecule_cache = {}
        self._archive_dir = os.path.join(cache_dir, "molecule_archives")
        self.poor_vina_threshold = float(poor_vina_threshold)
        self.max_consecutive_poor_vina = int(max_consecutive_poor_vina)
        assert self.max_consecutive_poor_vina > 0
        self._poor_vina_streak = 0
        self._best_scored_candidate = None
        # ===== 新增：集中管理上限 =====
        self.MAX_TITLES = 30            # 每次保存仅保留 Top-30 标题
        self.MAX_TOTAL_SKILLS = 100     # 全库洞见总量上限
        self.MAX_PER_TITLE = 5          # 每标题最多保留 5 条（你原逻辑）
    # ===== 新增：标题排序（按洞见条数降序，二级键用最近长度和，保证稳定性）=====
    def _rank_titles(self):
        def score(t):
            arr = self.skill_lib.get(t, [])
            # 主排序：洞见数量；辅排序：最近两条长度和（偏向最近有更新的标题）
            recent_two = arr[-2:] if len(arr) >= 2 else arr
            aux = sum(len(x) for x in recent_two)
            return (len(arr), aux)
        return sorted(self.skill_lib.keys(), key=score, reverse=True)
    # ===== 新增：全库裁剪，确保标题Top-30 & 全量≤100 =====
    def _prune_skill_lib(self):
        # 1) 按标题Top-30筛选
        keep_titles = set(self._rank_titles()[:self.MAX_TITLES])
        for t in list(self.skill_lib.keys()):
            if t not in keep_titles:
                del self.skill_lib[t]
        # 2) 每标题最多保留 MAX_PER_TITLE（保留最近的）
        for t in list(self.skill_lib.keys()):
            self.skill_lib[t] = self.skill_lib[t][-self.MAX_PER_TITLE:]
        # 3) 全库总量 ≤ MAX_TOTAL_SKILLS（优先保留最近；从“当前洞见较多的标题”的最旧项开始丢弃）
        def total_count():
            return sum(len(v) for v in self.skill_lib.values())
        while total_count() > self.MAX_TOTAL_SKILLS and self.skill_lib:
            # 找出当前洞见数最多的标题
            t = max(self.skill_lib.keys(), key=lambda k: len(self.skill_lib[k]))
            if not self.skill_lib[t]:
                del self.skill_lib[t]
                continue
            # 丢弃该标题最旧的一条（列表头部），保留最近
            self.skill_lib[t].pop(0)
            if len(self.skill_lib[t]) == 0:
                del self.skill_lib[t]
    def _merge_similar_titles(self):
        """
        合并 skill_lib 中相似的标题（key），将相似标题下的洞见合并到主标题下，删除冗余标题。
        两标题相似度大于0.75 视为相似。
        """
        titles = list(self.skill_lib.keys())
        merged = set()
        for title in titles:
            if title in merged:
                continue
            # 找到与当前标题相似的其他标题
            matches = get_close_matches(title, titles, n=len(titles), cutoff=0.5)
            for m in matches:
                if m != title and m in self.skill_lib:
                    # 合并洞见列表
                    self.skill_lib[title].extend(self.skill_lib[m])
                    # 标记并删除冗余标题
                    merged.add(m)
                    del self.skill_lib[m]
        # 合并后再去重
        self._dedup_insights()
    # ===== 新增：结构校验（最小侵入） =====
    def _is_valid_skill_lib(self, data):
        """
        要求结构为 {str: List[str]}。
        """
        if not isinstance(data, dict):
            return False
        for k, v in data.items():
            if not isinstance(k, str):
                return False
            if not isinstance(v, list):
                return False
            if not all(isinstance(x, str) for x in v):
                return False
        return True
    def _load_skill_lib(self):
        """
        读取技能库的自愈逻辑（仅最小改动）：
        - 正常读取并返回；
        - 若损坏或结构非法：若当前内存中有最新合法版本（self.skill_lib），用其覆盖损坏文件；
          否则重置为 {} 并写回。
        """
        if os.path.exists(self.skill_lib_file):
            try:
                with open(self.skill_lib_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self._is_valid_skill_lib(data):
                    return data
                # 结构非法，走自愈
            except Exception:
                # JSON损坏或读取异常，走自愈
                pass
        # 自愈：优先用内存中最新合法版本覆盖；否则重置为空
        if hasattr(self, "skill_lib") and self._is_valid_skill_lib(self.skill_lib) and self.skill_lib:
            healed = dict(self.skill_lib)
        else:
            healed = {}
        try:
            with open(self.skill_lib_file, "w", encoding="utf-8") as f:
                json.dump(healed, f, ensure_ascii=False, indent=4)
            print(f"⚠️ Skill library file corrupted or invalid. Auto-healed to valid state with {len(healed)} categories.")
        except Exception:
            # 写入异常则仍返回内存对象，避免中断
            pass
        return healed
    def _save_skill_lib(self):
        # 先合并磁盘中已有的内容
        existing = self._load_skill_lib()
        for title, insights in existing.items():
            if title in self.skill_lib:
                self.skill_lib[title].extend(insights)
            else:
                self.skill_lib[title] = insights
        # 合并相似标题 + 去重
        self._merge_similar_titles()
        self._dedup_insights()
        # === 关键新增：统一裁剪 ===
        self._prune_skill_lib()
        # 最终落盘
        with open(self.skill_lib_file, "w", encoding="utf-8") as f:
            json.dump(self.skill_lib, f, ensure_ascii=False, indent=4)
    def _load_history_logs(self):
        if os.path.exists(self.history_log_file):
            with open(self.history_log_file, "r") as f:
                return json.load(f)
        return []
    def _dedup_insights(self):
        """
        对每个标题下的洞见列表进行去重，保留首条出现内容。
        两条洞见相似度大于 0.4 视为重复。
        """
        for title, insights in list(self.skill_lib.items()):
            unique = []
            for ins in insights:
                if not any(SequenceMatcher(None, ins, u).ratio() > 0.4 for u in unique):
                    unique.append(ins)
            self.skill_lib[title] = unique
    def _load_history_logs(self):
        if os.path.exists(self.history_log_file):
            with open(self.history_log_file, "r") as f:
                return json.load(f)
        return []
    def _save_history_logs(self):
        with open(self.history_log_file, "w") as f:
            json.dump(self.history_logs, f, indent=4)
    def _update_generation_mode(self, response, source):
        """Parse a small Commander JSON decision; malformed output keeps mode."""
        text = str(response).rsplit("</think>", 1)[-1]
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                decision, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(decision, dict) or decision.get("mode") not in {"standard", "direct_large"}:
                continue
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                continue
            if source == "trend" and not isinstance(decision.get("analysis"), str):
                continue
            previous = self.generation_mode
            self.generation_mode = decision["mode"]
            self.mode_reason = reason[:1200]
            self.mode_history.append({"source": source, "iteration": self._mode_iteration,
                                      "previous_mode": previous, "mode": self.generation_mode,
                                      "reason": self.mode_reason})
            print(f"[MODE] {source}: {previous} -> {self.generation_mode}: {self.mode_reason}", flush=True)
            return decision
        print(f"[MODE] Invalid Commander mode JSON; keeping {self.generation_mode}", flush=True)
        return None

    def _mode_history_summary(self):
        """Enrich existing history with actual MW/HAC/LE; do not invent trials."""
        summary = []
        for row in self.history_logs[-10:]:
            try:
                candidate = (row["smiles"], float(row["vina"]), float(row["qed"]),
                             float(row["sa"]), float(row["composite"]))
                if not all(math.isfinite(v) for v in candidate[1:]) or candidate[1] >= 500:
                    continue
                metrics = self._candidate_metrics(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            summary.append({"smiles": row["smiles"], **metrics,
                            "mode_at_generation": row.get("generation_mode", "unknown"),
                            "source": row.get("source", "generated_or_resumed")})
        return summary

    def analyze_results(self, target_protein, prev_property, current_property):
        cot_prompt = f"""
        - **Target Protein**: {target_protein}
        - **Existing recorded history (initial candidates and round winners, not all trials)**: {json.dumps(self._mode_history_summary(), indent=4)}
        - **Current mode**: {self.generation_mode}; reason: {self.mode_reason}
        - **Earlier mode decisions**: {json.dumps(self.mode_history[-3:], indent=4)}
        - **Previous Property**: {prev_property}
        - **Current Property**: {current_property}
        """
        previous_knowledge = []
        for title, arr in self.skill_lib.items():
            for ins in arr:
                previous_knowledge.append(f"{title}: {ins}")
        prev_k_str = "\n".join(previous_knowledge)
        critique_response = self.commander.chat(
            cot_prompt,
            instruction=(
                "Analyze structural modifications on binding affinity and drug-like properties. "
                "Use the existing history and current mode to decide whether to keep or switch mode for the next iteration. "
                "standard uses 30 trials with MW<=500 then 20 with MW<=900. direct_large uses up to 50 trials "
                "targeting 500-800 Da from the first trial. Both preserve QED/SA filters and the existing final forced return. "
                "Do not favor larger molecules solely for more negative Vina; weigh LE, MW, QED and SA. "
                "Prefer standard when compact candidates progress or growth adds mass without adequate benefit. "
                "Consider direct_large when compact candidates stagnate and target-specific growth is plausible. "
                "History is selective, not a complete sampling timeline; do not invent failure rates or pocket contacts. "
                "Return JSON only: {\"analysis\":\"brief actionable critique\", "
                "\"mode\":\"standard or direct_large\", \"reason\":\"brief evidence-based rationale\"}."
            )
        )
        mode_decision = self._update_generation_mode(critique_response, "trend")
        critique = mode_decision["analysis"] if mode_decision else str(critique_response)
        title_c = self.commander.chat(
            critique,
            instruction="Generate a short title (3-5 words) summarizing this critique."
        ).strip()
        self.skill_lib.setdefault(title_c, []).append(critique)
        knowledge = self.commander.chat(
            f"Previous Knowledge:\n{prev_k_str}",
            instruction=(
                "Summarize key drug design insights or experience from past interactions in 20–30 words. "
                "Ensure the summary is constructive and self-contained."
            )
        )
        title_k = self.commander.chat(
            knowledge,
            instruction="Generate a short title (3-5 words) summarizing this knowledge."
        ).strip()
        self.skill_lib.setdefault(title_k, []).append(knowledge)
        insights = self.commander.chat(
            f"Critique: {critique}\nKnowledge: {knowledge}",
            instruction=(
                "Propose specific structural modifications for next-gen molecules in 20 words. "
                "Ensure suggestions are actionable and clear without extra context."
            )
        )
        intention = self.commander.chat(
            f"Prev: {prev_property}\nCurr: {current_property}\nInsights: {insights}\n",
            instruction="Inference next optimization intention in one concise, self-contained sentence:"
        )
        self._save_skill_lib()
        return insights, intention
    def retrive_knowledge(self, intention):
        # 汇总已有 memory
        memory = [ins for insights in self.skill_lib.values() for ins in insights]
        if len(memory) > 0:
            # 分步推理：选类别 -> 选最相关洞见 -> 摘要
            categories = list(self.skill_lib.keys())
            if intention:
                intention_prompt = f"Intention: {intention}. "
            else:
                intention_prompt = ""
            selected_title = self.commander.chat(
                f"{intention_prompt} Available categories: {categories}",
                instruction="Select the most relevant category for next optimization. Return the exact title."
            )
            matched_titles = get_close_matches(selected_title, categories, n=1, cutoff=0.4)
            if matched_titles:
                selected_title = matched_titles[0]
            # from pdb import set_trace
            # set_trace()
            insights_list = self.skill_lib.get(selected_title, [])
            if len(insights_list) > 1:
                sample_size = max(1, int(len(insights_list) * 0.1))
                insights_list = random.sample(insights_list, sample_size)
            memory_summary = self.commander.chat(
                f"Insight: {insights_list}",
                instruction="Summarize the given insights into one concise sentence."
            )
            print(f"📝 Memory Summary: {memory_summary}\n")
        else:
            memory_summary = ""
        return memory_summary
    def _save_new_molecule_to_csv(self, candidate, label):
        """Immediately append each newly docked unique molecule to cache/molecules.csv."""
        smiles, vina, qed, sa, composite = candidate
        if not all(math.isfinite(float(x)) for x in (vina, qed, sa, composite)):
            raise ValueError(f"Non-finite candidate metrics: {candidate}")
        # 500 is BindingSimulator's unsuccessful-docking sentinel.
        if float(vina) >= 500:
            return False
        key = self._candidate_key(candidate)
        if key in self._saved_seen_smiles:
            return False
        with open(self._improvement_csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([smiles, vina, qed, sa, composite])
            handle.flush()
            os.fsync(handle.fileno())
        self._saved_seen_smiles.add(key)
        self._scored_molecule_cache[key] = tuple(candidate)
        print(f"[MOLECULE SAVED] {label}: vina={vina:.3f} composite={composite:.4f}", flush=True)
        return True

    def _record_scored_candidate(self, candidate, label):
        """Save new valid molecules immediately; maintain Top-3 and improvement tracking."""
        smiles, vina, qed, sa, composite = candidate
        assert all(math.isfinite(float(x)) for x in (vina, qed, sa, composite))
        if float(vina) >= 500:
            return

        # Save regardless of improvement; exactly one primary CSV is written.
        self._save_new_molecule_to_csv(candidate, label)

        # Non-improving molecules remain eligible as Top-3 parents.
        self._elite_pool = self._select_top_candidates(
            self._elite_pool + [candidate], limit=3
        )
        if self._best_scored_candidate is None or composite > self._best_scored_candidate[4]:
            self._best_scored_candidate = candidate

        if composite <= self._saved_best_composite:
            print(f"[NO IMPROVEMENT] {label}: composite={composite:.4f}; "
                  f"saved_best={self._saved_best_composite:.4f}", flush=True)
            return

        self._saved_best_composite = composite
        self.molecule_list.append(list(candidate))
        # Improvement archive is retained; avoid writing the same molecule twice
        # to cache/molecules.csv.
        with open(self._archive_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([smiles, vina, qed, sa, composite, label])
            handle.flush()
            os.fsync(handle.fileno())
        print(f"[IMPROVEMENT SAVED] {label}: composite={composite:.4f}; "
              f"elite={len(self._elite_pool)}", flush=True)

    def _evaluate_generated_candidate(self, generation_prompt, protein_sequence, pdb_id, target_index, label, scaffold=False):
        """Return a scored candidate, or None for an exhausted/invalid generation."""
        try:
            mode_guidance = ("Prefer compact candidates; use the normal 30+20 procedure."
                             if self.generation_mode == "standard" else
                             "Explore a compact 500-800 Da ligand immediately. If the parent is small, "
                             "adapt the design objective to target-justified growth rather than pure shrinking.")
            generation_prompt += f"\nActive size policy: {mode_guidance} Reason: {self.mode_reason}"
            smiles = self.drug_expert.generate_molecule(
                generation_prompt, scaffold=scaffold, mode=self.generation_mode
            )
        except ValueError as exc:
            print(f"[WARN] {pdb_id} {label}: generation failed: {exc}", flush=True)
            return None
        if not isinstance(smiles, str) or not smiles.strip():
            print(f"[WARN] {pdb_id} {label}: invalid generated SMILES: {smiles!r}", flush=True)
            return None
        # Deduplicate BEFORE property calculation and expensive smina/Vina docking.
        # Cache is populated from this target's existing molecules.csv and new writes.
        key = self._candidate_key((smiles,))
        cached = self._scored_molecule_cache.get(key)
        if cached is not None:
            print(f"[CACHE HIT] {pdb_id} {label}: vina={cached[1]:.3f}; skipping docking", flush=True)
            # Return saved metrics for the normal downstream structural deduplication.
            # Do not record again or change the poor-Vina streak: no docking occurred.
            return cached

        props = self.drug_expert.compute_properties(smiles)
        if props is None:
            print(f"[WARN] {pdb_id} {label}: property calculation failed for {smiles!r}", flush=True)
            return None
        affinity = self.binding_simulator.CaculateAffinity(
            protein_seq=protein_sequence,
            drug_smiles=smiles,
            pdb_id=pdb_id,
            target_index=target_index,
        )["binding_affinity"]
        vina = float(affinity)
        qed = props['QuantitativeEstimateOfDruglikeness(↑)']
        sa = props['NormalizedSyntheticAccessibilityScore(↑)']
        composite = 0.5 * qed - 0.5 * vina + 0.4 * sa
        # Count only actual, finite docking scores. A good score resets the streak.
        if not math.isfinite(vina):
            raise ValueError(f"{pdb_id} {label}: non-finite Vina score: {vina}")
        candidate = (smiles, vina, qed, sa, composite)
        # Persist and compare BEFORE a round-level early stop can be raised.
        self._record_scored_candidate(candidate, label)
        if vina > self.poor_vina_threshold:
            self._poor_vina_streak += 1
            print(
                f"[VINA] {pdb_id} {label}: affinity={vina:.3f} > "
                f"{self.poor_vina_threshold:.2f}; consecutive poor "
                f"{self._poor_vina_streak}/{self.max_consecutive_poor_vina}",
                flush=True,
            )
            if self._poor_vina_streak >= self.max_consecutive_poor_vina:
                raise ConsecutivePoorVinaStop(
                    f"{pdb_id}: {self._poor_vina_streak} consecutive Vina scores "
                    f"> {self.poor_vina_threshold:.2f}"
                )
        else:
            self._poor_vina_streak = 0
        return candidate
    def run_drug_discovery_cycle(self, protein_sequence, protein_title="Unknown Protein", pdb_id="NotAvailable", target_index=None, protein_lookup_id=None, initial_elites=None):
        """Run all scheduled iterations; poor-Vina stopping applies to a round only."""
        self.generation_mode = "standard"
        self.mode_reason = "Default at start of this protein task"
        self.mode_history = []
        self._mode_iteration = 0
        self._poor_vina_streak = 0
        self._best_scored_candidate = None
        self._elite_pool = []
        self.molecule_list = []
        os.makedirs(self._archive_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self._improvement_csv_path)), exist_ok=True)
        self._saved_best_composite = -math.inf
        if not os.path.isfile(self._improvement_csv_path):
            with open(self._improvement_csv_path, "w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["SMILES", "vina", "qed", "sa", "composite"])

        # Load complete scored records once: O(1) canonical-SMILES lookups later.
        # Cache is target-local; docking scores cannot be shared across proteins.
        self._saved_seen_smiles = set()
        self._scored_molecule_cache = {}
        with open(self._improvement_csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"SMILES", "vina", "qed", "sa", "composite"}
            assert reader.fieldnames is not None and expected.issubset(reader.fieldnames), (
                f"Invalid molecules CSV header: {self._improvement_csv_path}"
            )
            for row in reader:
                candidate = (
                    row["SMILES"], float(row["vina"]), float(row["qed"]),
                    float(row["sa"]), float(row["composite"]),
                )
                assert all(math.isfinite(value) for value in candidate[1:]), (
                    f"Non-finite saved metrics in {self._improvement_csv_path}: {row}"
                )
                key = self._candidate_key(candidate)
                self._saved_seen_smiles.add(key)
                if key not in self._scored_molecule_cache:
                    self._scored_molecule_cache[key] = candidate
                self._saved_best_composite = max(self._saved_best_composite, candidate[4])
        print(f"[CACHE] Loaded {len(self._scored_molecule_cache)} unique scored molecules", flush=True)
        safe_target = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(pdb_id))
        run_id = uuid.uuid4().hex[:12]
        self._archive_path = os.path.join(self._archive_dir, f"{safe_target}_{target_index}_{run_id}.csv")
        with open(self._archive_path, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["SMILES", "vina", "qed", "sa", "composite", "source"])
        return self._run_drug_discovery_cycle_impl(
            protein_sequence=protein_sequence,
            protein_title=protein_title,
            pdb_id=pdb_id,
            target_index=target_index,
            protein_lookup_id=protein_lookup_id,
            initial_elites=initial_elites,
        )

    @staticmethod
    def _candidate_key(candidate):
        """Canonicalize SMILES for structural deduplication."""
        mol = Chem.MolFromSmiles(candidate[0])
        if mol is None:
            raise ValueError(f"Invalid candidate SMILES: {candidate[0]}")
        return Chem.MolToSmiles(mol, canonical=True)

    def _select_top_candidates(self, candidates, limit=3):
        """Select top distinct structures by composite score, with deterministic ties."""
        unique = {}
        for candidate in candidates:
            key = self._candidate_key(candidate)
            if key not in unique or candidate[4] > unique[key][4]:
                unique[key] = candidate
        return sorted(unique.values(), key=lambda x: x[4], reverse=True)[:limit]

    @staticmethod
    def _candidate_metrics(candidate):
        mol = Chem.MolFromSmiles(candidate[0])
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            raise ValueError("Cannot calculate size/LE for an invalid candidate")
        heavy_atoms = mol.GetNumHeavyAtoms()
        return {
            "binding_affinity": candidate[1],
            "QuantitativeEstimateOfDruglikeness": candidate[2],
            "NormalizedSyntheticAccessibilityScore": candidate[3],
            "MolecularWeight": float(Descriptors.MolWt(mol)),
            "HeavyAtomCount": heavy_atoms,
            "DockingLigandEfficiency": -float(candidate[1]) / heavy_atoms,
        }

    def _run_drug_discovery_cycle_impl(self, protein_sequence, protein_title="Unknown Protein", pdb_id="NotAvailable", target_index=None, protein_lookup_id=None, initial_elites=None):
        """Initialize diverse scaffolds, then evolve a three-parent elite pool."""
        receptor, reference = self.binding_simulator._resolve_inputs(
            pdb_id=pdb_id,
            target_index=target_index,
            receptor_path=None,
            reference_ligand_path=None,
        )
        print(f"CadreAI receptor: {receptor}")
        print(f"Docking box reference: {reference}")
        intention = ""
        print("Retrieving protein knowledge...")
        protein_knowledge = self.protein_expert.retrieval_interaction(protein_sequence)
        # One initial Commander call when reading protein information.
        try:
            mode_response = self.commander.chat(
                f"Protein information: {protein_knowledge}. Protein title: {protein_title}. "
                f"Existing recorded candidate history: {json.dumps(self._mode_history_summary())}",
                instruction=(
                    "Choose the initial generation mode for this protein. Return JSON only: "
                    "{\"mode\":\"standard or direct_large\", \"reason\":\"brief rationale\"}. "
                    "standard means compact-first: 30 trials MW<=500, then 20 trials MW<=900. "
                    "direct_large immediately explores 500-800 Da with the same 50-trial maximum. "
                    "Use standard when information is limited. Consider direct_large only when explicit "
                    "ligand/structural information or observed history gives a plausible reason to explore larger ligands. "
                    "Protein function or name alone does not establish pocket size. Do not invent contacts or affinity. "
                    "Weigh ligand efficiency and mass cost; do not choose growth solely for better Vina."
                ),
            )
            self._update_generation_mode(mode_response, "protein_info")
        except Exception as exc:
            print(f"[MODE] Initial decision failed; keeping standard: {exc}", flush=True)
        print(f"Protein Interaction Data: {protein_knowledge}\n")

        # Soft molecular-size targets for 25 generated initial candidates.
        # These are requests to the generator, not rigid filters on actual output.
        size_specs = [
            # Very small: suitable for small binding pockets
            ("very small polar fragment", 6, 10),
            ("very small polar fragment", 6, 10),
            ("small oxygen-rich scaffold", 9, 13),
            ("small oxygen-rich scaffold", 9, 13),
            ("small hydrogen-bonding scaffold", 10, 15),
            ("small hydrogen-bonding scaffold", 10, 15),
            ("small saturated heterocyclic scaffold", 10, 17),

            # Original: very small
            ("very small fragment-like scaffold", 12, 17),
            ("very small fragment-like scaffold", 12, 17),

            # Original: small
            ("small drug-like heterocyclic scaffold", 18, 22),
            ("small rigid aromatic scaffold", 18, 22),
            ("small saturated heterocyclic scaffold", 18, 22),
            ("small polar bioactive scaffold", 18, 22),

            # Original: medium-small
            ("medium-small nitrogen heterocycle", 23, 27),
            ("medium-small oxygen heterocycle", 23, 27),
            ("medium-small bicyclic scaffold", 23, 27),
            ("medium-small balanced lipophilic scaffold", 23, 27),

            # Original: medium
            ("medium rigid bicyclic scaffold", 28, 32),
            ("medium fused heteroaromatic scaffold", 28, 32),
            ("medium saturated ring-rich scaffold", 28, 32),

            # Original: medium-large
            ("medium-large compact heterocyclic scaffold", 33, 37),
            ("medium-large balanced polar scaffold", 33, 37),

            # Original: large
            ("large compact bioactive scaffold", 38, 42),

            # Original: alternative chemotype
            ("alternative novel synthetically accessible chemotype", 18, 35),
            ("alternative compact heteroaromatic chemotype", 20, 30),
        ]
        if self.generation_mode == "direct_large":
            # Keep the same scaffold-request count; align size hints with the selected mode.
            size_specs = [("compact larger bioactive scaffold", 35 + 5 * (i % 3), 42 + 5 * (i % 3))
                          for i in range(len(size_specs))]
        if initial_elites is None:
            initial_candidates = []
            seen_initial = set()

            for i, (size_label, low, high) in enumerate(size_specs):
                scaffold_advice = self.commander.chat(
                    f"Target Protein Info: {protein_knowledge} (Protein name: {protein_title}). "
                    f"Request a {size_label} scaffold with approximately {low}-{high} heavy atoms. "
                    f"Prior generated scaffolds: {[x[0] for x in initial_candidates]}",
                    instruction=(
                        "Suggest a chemically feasible scaffold that differs in ring system, "
                        "topology, and molecular size from prior scaffolds. "
                        "Respect the requested heavy-atom range and drug-like properties. "
                        "Do not simply add alkyl chains to change size."
                    ),
                )
                try:
                    candidate = self._evaluate_generated_candidate(
                        f"Generate a novel drug-like molecule for {protein_title}. "
                    f"Target size: {size_label}, approximately {low}-{high} heavy atoms. "
                    f"Use a distinct core scaffold and different ring system from these existing "
                    f"molecules: {[x[0] for x in initial_candidates]}. "
                    f"Keep the structure synthetically plausible. Design guidance: {scaffold_advice}",
                    protein_sequence, pdb_id, target_index,
                        f"stage=0 candidate={i + 1}/{len(size_specs)}", scaffold=True,
                    )
                except ConsecutivePoorVinaStop as exc:
                    print(f"[ROUND STOP] Initialization: {exc}; retaining scored candidates.", flush=True)
                    self._poor_vina_streak = 0
                    break
                if candidate is None:
                    continue
                key = self._candidate_key(candidate)
                if key in seen_initial:
                    print(f"[SKIP] Duplicate initial scaffold: {candidate[0]}", flush=True)
                    continue
                seen_initial.add(key)
                initial_candidates.append(candidate)
                mol = Chem.MolFromSmiles(candidate[0])
                print(f"[INIT] {i + 1}/{len(size_specs)} size={size_label} "
                      f"heavy_atoms={mol.GetNumHeavyAtoms()} MW={Descriptors.MolWt(mol):.1f} "
                      f"composite={candidate[4]:.4f}", flush=True)
                self.history_logs.append({
                    "smiles": candidate[0], "vina": candidate[1],
                    "qed": candidate[2], "sa": candidate[3],
                    "composite": candidate[4], "generation_mode": self.generation_mode,
                })

            if not self._elite_pool:
                raise RuntimeError(f"{pdb_id}: no valid initial candidates among {len(size_specs)} requests")

        else:
            assert initial_elites, "Cannot continue without initial elites"
            assert all(len(x) == 5 for x in initial_elites), "Each elite must be (SMILES,vina,qed,sa,composite)"
            self._elite_pool = self._select_top_candidates([tuple(x) for x in initial_elites], limit=3)
            self._best_scored_candidate = self._elite_pool[0]
            for candidate in self._elite_pool:
                self.history_logs.append({"smiles": candidate[0], "vina": candidate[1], "qed": candidate[2], "sa": candidate[3], "composite": candidate[4]})
            print(f"[RESUME] Loaded {len(self._elite_pool)} pre-scored elite parents; scaffold initialization skipped.", flush=True)

        elite_pool = list(self._elite_pool)
        best_ever = elite_pool[0]
        previous_metrics = self._candidate_metrics(best_ever)
        print(f"[INIT] Selected {len(elite_pool)} unique parents:")
        for rank, parent in enumerate(elite_pool, 1):
            print(f"  Parent {rank}: composite={parent[4]:.4f} SMILES={parent[0]}")

        # Preserve the original nested search: 3 elite parents x 3 design variants.
        # The different objectives are soft constraints, not strict filters.
        # design_objectives = (
        #     (
        #         "grow_and_engage",
        #         "Explore a modest, purposeful molecular expansion: grow into a plausible "
        #         "unoccupied pocket region by adding or extending a compact ring or "
        #         "substituent; introduce interactions without excessive lipophilicity. "
        #         "Prefer a molecular-size increase, if chemically justified.",
        #     ),
        #     (
        #         "trim_and_optimize",
        #         "Explore a smaller, more efficient analogue: remove or contract a "
        #         "bulky appendage, simplify a ring/substituent, or reduce rotatable "
        #         "bonds while preserving the binding pharmacophore. Prefer fewer "
        #         "heavy atoms and improved ligand efficiency.",
        #     ),
        #     (
        #         "scaffold_or_polarity_change",
        #         "Explore a qualitatively different modification: heteroatom exchange, "
        #         "bioisosteric replacement, ring topology change, or relocation of "
        #         "hydrogen-bond donors/acceptors. Avoid simply growing or trimming "
        #         "the molecule; prioritize chemical novelty and viable interactions.",
        #     ),
        # )

        # design_objectives = (
        #     (
        #         "grow_and_engage",
        #         "Expand the molecule moderately into unexplored pocket regions. "
        #         "Introduce compact rings or substituents to improve binding interactions "
        #         "while maintaining drug-likeness and synthetic accessibility.",
        #     ),
        #     (
        #         "trim_and_optimize",
        #         "Reduce molecular size and complexity by removing unnecessary groups, "
        #         "simplifying ring systems, or reducing rotatable bonds. "
        #         "Preserve essential pharmacophoric features and improve ligand efficiency.",
        #     ),
        #     (
        #         "scaffold_or_polarity_change",
        #         "Explore bioisosteric replacements, heteroatom substitutions, "
        #         "ring topology changes, or alternative hydrogen-bond patterns. "
        #         "Prioritize chemical novelty and plausible binding interactions.",
        #     ),
        #     (
        #         "rigidify_and_preorganize",
        #         "Reduce conformational flexibility through ring closure, "
        #         "linker rigidification, or conformational constraints. "
        #         "Preserve key interaction sites and avoid excessive ring strain "
        #         "or synthetic complexity.",
        #     ),
        #     (
        #         "interaction_and_solubility_balance",
        #         "Optimize the balance between hydrophobic contacts, hydrogen bonding, "
        #         "and aqueous solubility. Explore strategic substituent replacements, "
        #         "heteroatom positioning, and polarity redistribution while maintaining "
        #         "reasonable molecular weight and synthetic accessibility.",
        #     ),
        # )
        design_objectives = (
            (
                "grow_and_engage",
                "Expand the molecule moderately into unexplored pocket regions. "
                "Introduce compact rings or substituents to improve binding interactions "
                "while maintaining drug-likeness and synthetic accessibility.",
            ),
            (
                "trim_and_optimize",
                "Reduce molecular size and complexity by removing unnecessary groups, "
                "simplifying ring systems, or reducing rotatable bonds. "
                "Preserve essential pharmacophoric features and improve ligand efficiency.",
            ),
            (
                "radical_scaffold_hopping",
                "Perform a radical molecular redesign to escape the current local optimum. "
                "You are explicitly allowed to abandon the existing molecular scaffold, "
                "ring topology, linker architecture, and substituent arrangement. "
                "Explore fundamentally different chemotypes, including alternative "
                "heterocyclic systems, fused or bridged rings, and distinct molecular "
                "shapes or pharmacophore arrangements. "
                "Do not restrict the redesign to incremental modifications or close "
                "structural analogues of the parent molecule. "
                "Retain only those molecular features that are strongly justified "
                "by the target binding environment. "
                "Prioritize discovering a substantially different, chemically valid "
                "molecule with the potential to achieve a markedly stronger docking "
                "affinity than the current elite.",
            ),
            (
                "rigidify_and_preorganize",
                "Reduce conformational flexibility through ring closure, "
                "linker rigidification, or conformational constraints. "
                "Preserve key interaction sites and avoid excessive ring strain "
                "or synthetic complexity.",
            ),
            (
                "interaction_and_solubility_balance",
                "Optimize the balance between hydrophobic contacts, hydrogen bonding, "
                "and aqueous solubility. Explore strategic substituent replacements, "
                "heteroatom positioning, and polarity redistribution while maintaining "
                "reasonable molecular weight and synthetic accessibility.",
            ),
        )
        for iteration in range(1, self.max_iterations + 1):
            self._mode_iteration = iteration
            print(f"\nIteration {iteration} begins (elite={len(elite_pool)})...")
            memory_summary = self.retrive_knowledge(intention)
            offspring = []
            round_stopped = False
            seen_offspring = {self._candidate_key(x) for x in elite_pool}
            for parent_idx, parent in enumerate(elite_pool, start=1):
                parent_smiles = parent[0]
                parent_metrics = self._candidate_metrics(parent)
                sibling_smiles = []
                other_elites = [x[0] for x in elite_pool if x is not parent]
                for variant_idx, (objective_name, objective) in enumerate(design_objectives, start=1):
                    print(
                        f"[SEARCH] iteration={iteration} parent={parent_idx}/{len(elite_pool)} "
                        f"variant={variant_idx}/{len(design_objectives)} objective={objective_name}",
                        flush=True,
                    )
                    suggestion = self.commander.chat(
                        f"Protein: {protein_knowledge}. Parent SMILES: {parent_smiles}. "
                        f"Parent properties: {parent_metrics}. "
                        f"Other elite parents: {other_elites}. "
                        f"Previously generated variants from this parent: {sibling_smiles}. "
                        f"Memory: {memory_summary}. Global intention: {intention}. "
                        f"Variant objective ({variant_idx}/{len(design_objectives)}): {objective}",
                        instruction=(
                            "Suggest concrete, chemically valid modifications for this "
                            "specific design objective. Keep the parent as the starting "
                            "point, but explore a distinct structural direction from "
                            "the other variants. Address binding, QED and SA tradeoffs. "
                            "Return actionable modification guidance, not a SMILES."
                        ),
                    )
                    try:
                        candidate = self._evaluate_generated_candidate(
                            f"Optimize the parent SMILES {parent_smiles} for {protein_title}. "
                        f"Design objective: {objective} "
                        f"Modification guidance: {suggestion}. "
                        f"Avoid repeating these sibling variants: {sibling_smiles}. "
                        f"Create one chemically valid, drug-like modified SMILES "
                        f"with a distinct structural change from the parent.",
                        protein_sequence, pdb_id, target_index,
                        f"iteration={iteration} parent={parent_idx}/{len(elite_pool)} "
                        f"variant={variant_idx}/{len(design_objectives)} objective={objective_name}",
                        )
                    except ConsecutivePoorVinaStop as exc:
                        print(f"[ROUND STOP] iteration={iteration}: {exc}; continuing next iteration.", flush=True)
                        round_stopped = True
                        self._poor_vina_streak = 0
                        break
                    if candidate is None:
                        continue
                    key = self._candidate_key(candidate)
                    if key in seen_offspring:
                        print(
                            f"[SKIP] Duplicate/unchanged offspring: {candidate[0]}",
                            flush=True,
                        )
                        continue
                    seen_offspring.add(key)
                    sibling_smiles.append(candidate[0])
                    offspring.append(candidate)
                    print(
                        f"[OFFSPRING] objective={objective_name} "
                        f"composite={candidate[4]:.4f} vina={candidate[1]:.3f} "
                        f"SMILES={candidate[0]}", flush=True,
                    )
                if round_stopped:
                    break
            elite_pool = list(self._elite_pool)
            best_ever = elite_pool[0]
            if not offspring:
                print(f"[WARN] {pdb_id}: iteration {iteration} produced no valid offspring; retaining elites.", flush=True)
                previous_metrics = self._candidate_metrics(best_ever)
                continue

            round_winner = max(offspring, key=lambda x: x[4])
            self.history_logs.append({
                "smiles": round_winner[0], "vina": round_winner[1],
                "qed": round_winner[2], "sa": round_winner[3],
                "composite": round_winner[4], "generation_mode": self.generation_mode,
            })
            elite_pool = list(self._elite_pool)
            best_ever = elite_pool[0]
            current_metrics = self._candidate_metrics(best_ever)
            print(f"[ITER {iteration}] Retained top-{len(elite_pool)} distinct molecules:")
            for rank, member in enumerate(elite_pool, 1):
                print(f"  #{rank}: composite={member[4]:.4f}, vina={member[1]:.3f}, SMILES={member[0]}")
            insights, intention = self.analyze_results(
                target_protein=protein_title,
                prev_property=previous_metrics,
                current_property=current_metrics,
            )
            print(f"New insight: {insights}")
            previous_metrics = dict(current_metrics)

        self._save_history_logs()
        print(f"\nImproved molecules saved: {len(self.molecule_list)}")
        return {
            "final_smiles": best_ever[0],
            "molecule_list": self.molecule_list,
            "elite_pool": [list(candidate) for candidate in self._elite_pool],
            "molecule_archive_path": self._archive_path,
            "improvement_csv_path": self._improvement_csv_path,
            "generation_mode": self.generation_mode,
            "mode_history": list(self.mode_history),
        }
