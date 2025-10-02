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

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')


class DrugVoyager:
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
        need_save_each_step=False
    ):
        # wandb.init(
        #     project="DrugVoyager", name="drug_discovery_with_deepseek_commander",
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
            out_dir=cache_dir + f"/receptor{description}/",
            use_msa_server=use_msa_server,
            accelerator=accelerator,
            need_cache=False
        )
        # 使用官方 DeepSeek 实例
        self.commander = Deepseek_official(key="sk-35965cbdd484474bb51cb8d30e6655c3", cache_dir=os.path.join(cache_dir, "jsons/"))
        self.max_iterations = max_iterations
        self.skill_lib_file = cache_dir + skill_lib_file
        self.history_log_file = cache_dir + history_log_file
        # 技能库改为标题->洞见列表的字典
        self.skill_lib = self._load_skill_lib()
        self.history_logs = self._load_history_logs()
        self.shift_nochange_round = 5
        self.molecule_list = []

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

    # def _load_skill_lib(self):
        if os.path.exists(self.skill_lib_file):
            with open(self.skill_lib_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

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


    def analyze_results(self, target_protein, prev_property, current_property):
        cot_prompt = f"""
        - **Target Protein**: {target_protein}
        - **Molecular Optimization history**: {json.dumps(self.history_logs[-3:], indent=4)}
        - **Previous Property**: {prev_property}
        - **Current Property**: {current_property}
        """
        previous_knowledge = []
        for title, arr in self.skill_lib.items():
            for ins in arr:
                previous_knowledge.append(f"{title}: {ins}")
        prev_k_str = "\n".join(previous_knowledge)

        critique = self.commander.chat(
            cot_prompt,
            instruction=(
                "Analyze structural modifications on binding affinity and drug-like properties. "
                "Provide a constructive, self-contained insight understandable without background."
            )
        )
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

    def run_drug_discovery_cycle(self, protein_sequence, protein_title="Unknown Protein", pdb_id="NotAvailable"):
        
        intention = ""
        print("🔍 Retrieving protein knowledge...")
        protein_knowledge = self.protein_expert.retrieval_interaction(protein_sequence)
        print(f"📌 Protein Interaction Data: {protein_knowledge}\n")
        # 初始三组候选分子生成（首次无 memory 指引）
        candidates = []
        for i in range(8):
            prompt = self.commander.chat(
                f"Target Protein Info: {protein_knowledge} (Protein name: {protein_title})",
                instruction=(
                    "Suggest a small molecule scaffold. For target protein drug design."
                )
            )
            smiles = self.drug_expert.generate_molecule(f"Generate a drug for {protein_title}, the drug should: {prompt}")
            props = self.drug_expert.compute_properties(smiles)
            affinity = self.binding_simulator.CaculateAffinity(
                protein_sequence, smiles, pdb_id=pdb_id
            ).get('binding_affinity', 500.0)
            vina = float(affinity)
            qed = props.get('QuantitativeEstimateOfDruglikeness(↑)', 0.0)
            sa = props.get('NormalizedSyntheticAccessibilityScore(↑)', 0.0)
            composite = 0.5 * qed - 0.5 * vina + 0.4 * sa
            candidates.append((smiles, vina, qed, sa, composite))
            self.history_logs.append({
                "smiles": smiles,
                "vina": vina,
                "qed": qed,
                "sa": sa,
                "composite": composite
            })
        # from pdb import set_trace
        # set_trace()
        current_smiles, vina, qed, sa, composite = max(candidates, key=lambda x: x[4])
        print(f"🎯 Selected Best Initial Candidate: {current_smiles}\n")
        self.molecule_list.append([current_smiles, vina, qed, sa, composite])
        previous_metrics = {
            "binding_affinity": vina,
            "QuantitativeEstimateOfDruglikeness": qed,
            "NormalizedSyntheticAccessibilityScore": sa
        }
        current_metrics = previous_metrics

        # 迭代
        insights = None
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🚀 Iteration {iteration} begins...")
            memory_summary = self.retrive_knowledge(intention)
            candidate_round = []
            for i in range(3):
                # 在 prompt 中加入 memory_summary
                prompt = self.commander.chat(
                    (
                        f"Protein: {protein_knowledge}. Prev SMILES: {current_smiles}, Props: {current_metrics}. "
                        f"Insights summary :{memory_summary}."
                        f"Intention: {intention}."
                        f"Your Suggestion:"
                    ),
                    instruction=(
                        "Based on this summarized insights and knowledge, suggest structural modifications for optimization. "
                        "Focus on weight/substructure changes."
                    )
                )
                print(f"===work with : {pdb_id}===\n")
                smiles = self.drug_expert.generate_molecule(
                    f"Based on {current_smiles} and suggestions: {prompt}, generate SMILES"
                )
                props = self.drug_expert.compute_properties(smiles)
                # toxicity = self.drug_expert.compute_toxicity(smiles)
                
                affinity = self.binding_simulator.CaculateAffinity(
                    protein_sequence, smiles, pdb_id=pdb_id
                ).get('binding_affinity', 500.0)
                vina = float(affinity)
                qed = props.get('QuantitativeEstimateOfDruglikeness(↑)', 0.0)
                sa = props.get('NormalizedSyntheticAccessibilityScore(↑)', 0.0)
                composite = 0.5 * qed - 0.5 * vina + 0.4 * sa
                candidate_round.append((smiles, vina, qed, sa, composite))

            current_smiles, vina, qed, sa, composite = max(candidate_round, key=lambda x: x[4])
            current_metrics = {
                "binding_affinity": vina,
                "QuantitativeEstimateOfDruglikeness": qed,
                "NormalizedSyntheticAccessibilityScore": sa
            }
            self.history_logs.append({
                "smiles": smiles,
                "vina": vina,
                "qed": qed,
                "sa": sa,
                "composite": composite
            })
            print(f"🎯 Iteration {iteration} Winner: {current_smiles}\n")
            print(f"===work with : {pdb_id}===\n")
            self.molecule_list.append([current_smiles, vina, qed, sa, composite])
            insights, intention = self.analyze_results(
                target_protein=protein_title,
                prev_property=previous_metrics,
                current_property=current_metrics
            )
            print(f"💡 New Insight: {insights}\n")
            print(f"===work with : {pdb_id}===\n")
            previous_metrics =  current_metrics
        self._save_history_logs()
        print(f"\n✅ Multi-iteration Molecule Evolution Logged! Total iterations: {len(self.molecule_list)}")
        return {"final_smiles": current_smiles, "molecule_list": self.molecule_list}


import tempfile

class DrugVoyagerPro:
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
        need_save_each_step=False
    ):
        # wandb.init(
        #     project="DrugVoyager", name="drug_discovery_with_deepseek_commander",
        #     config={"device": device, "max_iterations": max_iterations}
        # )

        os.makedirs(cache_dir, exist_ok=True)

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
            out_dir=cache_dir + f"/receptor{description}/",
            use_msa_server=use_msa_server,
            accelerator=accelerator,
            need_cache=False
        )
        # 使用官方 DeepSeek 实例
        self.commander = Deepseek_official(key="sk-35965cbdd484474bb51cb8d30e6655c3",
                                           cache_dir=os.path.join(cache_dir, "jsons/"))
        self.max_iterations = max_iterations
        self.cache_dir = cache_dir
        self.skill_lib_file = os.path.join(cache_dir, skill_lib_file)
        self.history_log_file = os.path.join(cache_dir, history_log_file)
        self.molecules_csv_file = os.path.join(cache_dir, "molecules.csv")
        self.need_save_each_step = need_save_each_step

        # 技能库改为标题->洞见列表的字典
        self.skill_lib = self._load_skill_lib()
        self.history_logs = self._load_history_logs()
        self.shift_nochange_round = 5
        self.molecule_list = []

        # ===== 集中管理上限 =====
        self.MAX_TITLES = 30            # 每次保存仅保留 Top-30 标题
        self.MAX_TOTAL_SKILLS = 100     # 全库洞见总量上限
        self.MAX_PER_TITLE = 5          # 每标题最多保留 5 条

        # 初始化 CSV 头
        self._ensure_molecule_csv_header()

    # ---------- 持久化与原子写辅助 ----------
    def _atomic_write_json(self, path: str, data_obj):
        """原子写 JSON：写入临时文件，再替换目标文件，避免中间态损坏。"""
        dirpath = os.path.dirname(path) or "."
        os.makedirs(dirpath, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=dirpath, suffix=".tmp", encoding="utf-8") as tf:
            json.dump(data_obj, tf, ensure_ascii=False, indent=4)
            tmp_name = tf.name
        os.replace(tmp_name, path)  # 原子替换（同分区）

    def _append_history(self, record: dict):
        """追加历史记录并立刻落盘（原子写）。"""
        self.history_logs.append(record)
        # 即时落盘，防崩溃丢失
        self._atomic_write_json(self.history_log_file, self.history_logs)

    def _ensure_molecule_csv_header(self):
        """确保 molecules.csv 有表头。"""
        if not os.path.exists(self.molecules_csv_file):
            with open(self.molecules_csv_file, "w", encoding="utf-8") as f:
                f.write("SMILES,Vina,QED,SA,Composite\n")

    def _append_molecule_csv(self, smiles, vina, qed, sa, composite):
        """即时把 winner 追加到 CSV（避免崩溃丢失）。"""
        with open(self.molecules_csv_file, "a", encoding="utf-8") as f:
            f.write(f"{smiles},{vina},{qed},{sa},{composite}\n")

    # ---------- 技能库 ----------
    def _rank_titles(self):
        def score(t):
            arr = self.skill_lib.get(t, [])
            recent_two = arr[-2:] if len(arr) >= 2 else arr
            aux = sum(len(x) for x in recent_two)
            return (len(arr), aux)
        return sorted(self.skill_lib.keys(), key=score, reverse=True)

    def _prune_skill_lib(self):
        keep_titles = set(self._rank_titles()[:self.MAX_TITLES])
        for t in list(self.skill_lib.keys()):
            if t not in keep_titles:
                del self.skill_lib[t]
        for t in list(self.skill_lib.keys()):
            self.skill_lib[t] = self.skill_lib[t][-self.MAX_PER_TITLE:]

        def total_count():
            return sum(len(v) for v in self.skill_lib.values())

        while total_count() > self.MAX_TOTAL_SKILLS and self.skill_lib:
            t = max(self.skill_lib.keys(), key=lambda k: len(self.skill_lib[k]))
            if not self.skill_lib[t]:
                del self.skill_lib[t]
                continue
            self.skill_lib[t].pop(0)
            if len(self.skill_lib[t]) == 0:
                del self.skill_lib[t]

    def _merge_similar_titles(self):
        titles = list(self.skill_lib.keys())
        merged = set()
        for title in titles:
            if title in merged:
                continue
            matches = get_close_matches(title, titles, n=len(titles), cutoff=0.5)
            for m in matches:
                if m != title and m in self.skill_lib:
                    self.skill_lib[title].extend(self.skill_lib[m])
                    merged.add(m)
                    del self.skill_lib[m]
        self._dedup_insights()

    def _is_valid_skill_lib(self, data):
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
        if os.path.exists(self.skill_lib_file):
            try:
                with open(self.skill_lib_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self._is_valid_skill_lib(data):
                    return data
            except Exception:
                pass
        # 自愈：空库写回
        healed = {}
        try:
            self._atomic_write_json(self.skill_lib_file, healed)
            print(f"⚠️ Skill library file corrupted or invalid. Auto-healed to valid state with 0 categories.")
        except Exception:
            pass
        return healed

    def _save_skill_lib(self):
        existing = self._load_skill_lib()
        for title, insights in existing.items():
            if title in self.skill_lib:
                self.skill_lib[title].extend(insights)
            else:
                self.skill_lib[title] = insights
        self._merge_similar_titles()
        self._dedup_insights()
        self._prune_skill_lib()
        self._atomic_write_json(self.skill_lib_file, self.skill_lib)

    def _load_history_logs(self):
        if os.path.exists(self.history_log_file):
            try:
                with open(self.history_log_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history_logs(self):
        # 仍保留该方法（兼容旧调用），但实际每次 _append_history 已即时落盘
        self._atomic_write_json(self.history_log_file, self.history_logs)

    def _dedup_insights(self):
        for title, insights in list(self.skill_lib.items()):
            unique = []
            for ins in insights:
                if not any(SequenceMatcher(None, ins, u).ratio() > 0.4 for u in unique):
                    unique.append(ins)
            self.skill_lib[title] = unique

    # ---------- 分析/记忆 ----------
    def analyze_results(self, target_protein, prev_property, current_property):
        cot_prompt = f"""
        - **Target Protein**: {target_protein}
        - **Molecular Optimization history**: {json.dumps(self.history_logs[-3:], indent=4)}
        - **Previous Property**: {prev_property}
        - **Current Property**: {current_property}
        """
        previous_knowledge = []
        for title, arr in self.skill_lib.items():
            for ins in arr:
                previous_knowledge.append(f"{title}: {ins}")
        prev_k_str = "\n".join(previous_knowledge)

        critique = self.commander.chat(
            cot_prompt,
            instruction=(
                "Analyze structural modifications on binding affinity and drug-like properties. "
                "Provide a constructive, self-contained insight understandable without background."
            )
        )
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
        memory = [ins for insights in self.skill_lib.values() for ins in insights]
        if len(memory) > 0:
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

    # ---------- 主流程 ----------
    def _maybe_record_molecule(self, smiles, vina, qed, sa, composite, force: bool = False):
        """按开关记录到内存，并在需要时即时写入 CSV。"""
        if self.need_save_each_step or force:
            self.molecule_list.append([smiles, vina, qed, sa, composite])
            self._append_molecule_csv(smiles, vina, qed, sa, composite)

    def run_drug_discovery_cycle(self, protein_sequence, protein_title="Unknown Protein", pdb_id="NotAvailable"):
        intention = ""
        print("🔍 Retrieving protein knowledge...")
        protein_knowledge = self.protein_expert.retrieval_interaction(protein_sequence)
        print(f"📌 Protein Interaction Data: {protein_knowledge}\n")

        # 初始候选生成
        candidates = []
        for i in range(8):
            prompt = self.commander.chat(
                f"Target Protein Info: {protein_knowledge} (Protein name: {protein_title})",
                instruction=("Suggest a small molecule scaffold. For target protein drug design.")
            )
            smiles = self.drug_expert.generate_molecule(
                f"Generate a drug for {protein_title}, the drug should: {prompt}"
            )
            props = self.drug_expert.compute_properties(smiles)
            affinity = self.binding_simulator.CaculateAffinity(
                protein_sequence, smiles, pdb_id=pdb_id
            ).get('binding_affinity', 500.0)
            vina = float(affinity)
            qed = props.get('QuantitativeEstimateOfDruglikeness(↑)', 0.0)
            sa = props.get('NormalizedSyntheticAccessibilityScore(↑)', 0.0)
            composite = 0.5 * qed - 0.5 * vina + 0.4 * sa

            candidates.append((smiles, vina, qed, sa, composite))
            # 历史日志：立即落盘
            self._append_history({
                "stage": "init",
                "smiles": smiles, "vina": vina, "qed": qed, "sa": sa, "composite": composite
            })

        # 初始最佳
        current_smiles, vina, qed, sa, composite = max(candidates, key=lambda x: x[4])
        print(f"🎯 Selected Best Initial Candidate: {current_smiles}\n")
        # 可选：记录初始最佳到 CSV
        self._maybe_record_molecule(current_smiles, vina, qed, sa, composite, force=False)

        previous_metrics = {
            "binding_affinity": vina,
            "QuantitativeEstimateOfDruglikeness": qed,
            "NormalizedSyntheticAccessibilityScore": sa
        }
        current_metrics = dict(previous_metrics)

        # 迭代
        insights = None
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🚀 Iteration {iteration} begins...")
            memory_summary = self.retrive_knowledge(intention)
            candidate_round = []
            for i in range(3):
                prompt = self.commander.chat(
                    (f"Protein: {protein_knowledge}. Prev SMILES: {current_smiles}, Props: {current_metrics}. "
                     f"Insights summary :{memory_summary}. Intention: {intention}. Your Suggestion:"),
                    instruction=("Based on this summarized insights and knowledge, suggest structural modifications for optimization. "
                                 "Focus on weight/substructure changes.")
                )
                print(f"===work with : {pdb_id}===\n")
                smi_i = self.drug_expert.generate_molecule(
                    f"Based on {current_smiles} and suggestions: {prompt}, generate SMILES"
                )
                props_i = self.drug_expert.compute_properties(smi_i)
                affinity_i = self.binding_simulator.CaculateAffinity(
                    protein_sequence, smi_i, pdb_id=pdb_id
                ).get('binding_affinity', 500.0)
                vina_i = float(affinity_i)
                qed_i = props_i.get('QuantitativeEstimateOfDruglikeness(↑)', 0.0)
                sa_i = props_i.get('NormalizedSyntheticAccessibilityScore(↑)', 0.0)
                comp_i = 0.5 * qed_i - 0.5 * vina_i + 0.4 * sa_i
                candidate_round.append((smi_i, vina_i, qed_i, sa_i, comp_i))

            # 本轮 winner
            winner = max(candidate_round, key=lambda x: x[4])
            current_smiles, vina, qed, sa, composite = winner
            current_metrics = {
                "binding_affinity": vina,
                "QuantitativeEstimateOfDruglikeness": qed,
                "NormalizedSyntheticAccessibilityScore": sa
            }

            # 历史日志：立即落盘
            self._append_history({
                "stage": f"iter_{iteration}",
                "smiles": current_smiles, "vina": vina, "qed": qed, "sa": sa, "composite": composite
            })

            print(f"🎯 Iteration {iteration} Winner: {current_smiles}\n")
            print(f"===work with : {pdb_id}===\n")

            # winner 即时写 CSV（受 need_save_each_step 控制）
            self._maybe_record_molecule(current_smiles, vina, qed, sa, composite, force=False)

            insights, intention = self.analyze_results(
                target_protein=protein_title,
                prev_property=previous_metrics,
                current_property=current_metrics
            )
            print(f"💡 New Insight: {insights}\n")
            print(f"===work with : {pdb_id}===\n")
            previous_metrics = current_metrics

        # 最终分子：确保写入 CSV（即使 need_save_each_step=False）
        self._maybe_record_molecule(current_smiles, vina, qed, sa, composite, force=True)

        # 再次冗余保存一次历史（多一层保险）
        self._save_history_logs()

        print(f"\n✅ Multi-iteration Molecule Evolution Logged! Total iterations: {len(self.molecule_list)}")
        return {"final_smiles": current_smiles, "molecule_list": self.molecule_list}