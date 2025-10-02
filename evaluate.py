#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen  # LogP
from rdkit.Chem import Lipinski  # HBD/HBA

# ========= 基础 I/O =========
def parse_molecules_csv(path):
    """读取标准 CSV 文件并返回 DataFrame"""
    return pd.read_csv(path)

# ========= 分子描述符 & Lipinski =========
def compute_descriptors(smiles):
    """
    根据 SMILES 计算分子量、LogP、氢键供体数、氢键受体数
    返回: dict 或 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
    }

def compute_lipinski_score(desc):
    """
    Lipinski 4条规则合规计数:
    1) MolWt <= 500
    2) LogP <= 5
    3) HBD <= 5
    4) HBA <= 10
    """
    if not desc:
        return None
    score = 0
    score += 1 if desc["MolWt"] <= 500 else 0
    score += 1 if desc["LogP"] <= 5 else 0
    score += 1 if desc["NumHDonors"] <= 5 else 0
    score += 1 if desc["NumHAcceptors"] <= 10 else 0
    return score

# ========= 代表样本收集（已与最新版逻辑同步） =========
def collect_representatives(
    root_dir,
    *,
    remove_inactive=False,        # 与最新版对齐，默认不删除
    baseline_total=1767,          # 与最新版对齐
    strict=True                   # True=CrossDocked2020严筛；False=可切到“粗筛”
):
    """
    遍历 root_dir 下所有子目录，读取每个子目录的 molecules.csv，
    先按最新版逻辑进行阈值筛选（strict: vina<-8.18 & qed>0.25 & sa>0.59），
    在“合格集合”中取 composite 最大的记录作为代表；
    若不存在合格样本，记为 inactive（可选是否删除该 csv）。

    同时保留“需同步版”的功能：
    - 记录 PDB_ID
    - 计算并填充分子描述符（MolWt/LogP/HBD/HBA）
    - 计算 Lipinski 分数
    """
    inactive = 0
    total = 0
    reps = []

    root_dir = Path(root_dir)
    need_cols = {"vina", "qed", "sa", "composite"}  # 同步最新版最小必需列

    for sub in root_dir.iterdir():
        csv_path = sub / "molecules.csv"
        if not csv_path.is_file():
            continue

        df = parse_molecules_csv(csv_path)
        if not need_cols.issubset(df.columns):
            print(f"Skip (missing cols): {sub.name}")
            continue

        df = df.dropna(subset=list(need_cols))

        # —— 与最新版同步的筛选逻辑 ——
        if strict:
            eligible = df[(df["vina"] < -8.18) & (df["qed"] > 0.25) & (df["sa"] > 0.59)]
        else:
            # 你在最新版中给过“粗筛”示例，这里保留开关以便切换
            eligible = df[(df["vina"] < -6) & (df["qed"] > 0.45) & (df["sa"] > 0.4)]
            # 或者极宽松：eligible = df[(df['vina'] < -1)]

        if not eligible.empty:
            # 在合格集合中取 composite 最大
            idx = eligible["composite"].idxmax()
            row = eligible.loc[idx].copy()
            row["PDB_ID"] = sub.name  # 保留需同步版功能：记录来源目录为 PDB_ID

            # 计算描述符 & Lipinski（保留需同步版功能）
            smi_col = "SMILES" if "SMILES" in row.index else None
            if smi_col and isinstance(row[smi_col], str):
                desc = compute_descriptors(row[smi_col])
                if desc:
                    row["MolWt"] = desc["MolWt"]
                    row["LogP"] = desc["LogP"]
                    row["NumHDonors"] = desc["NumHDonors"]
                    row["NumHAcceptors"] = desc["NumHAcceptors"]
                    row["Lipinski"] = compute_lipinski_score(desc)
                else:
                    row["MolWt"] = row["LogP"] = row["NumHDonors"] = row["NumHAcceptors"] = row["Lipinski"] = None
            else:
                row["MolWt"] = row["LogP"] = row["NumHDonors"] = row["NumHAcceptors"] = row["Lipinski"] = None

            reps.append(row)

        else:
            # 不存在合格样本 → inactive
            best = df.loc[df["composite"].idxmax()] if not df.empty else None
            if best is not None:
                print(
                    f"Inactive (no eligible): {sub.name} | best -> "
                    f"vina={best.get('vina', None)}, qed={best.get('qed', None)}, sa={best.get('sa', None)}"
                )
            else:
                print(f"Inactive (empty df): {sub.name}")

            inactive += 1
            if remove_inactive:
                try:
                    os.remove(csv_path)
                except Exception as e:
                    print(f"Failed to remove {csv_path}: {e}")

        total += 1

    active = total - inactive
    activate_global_rate = (active / baseline_total) if baseline_total else 0.0
    activate_current_rate = (active / total) if total else 0.0

    print("Total:", total)
    print("Inactive:", inactive)
    print("Active:", active)
    print("Activate global rate:", activate_global_rate)
    print("Activate current rate:", activate_current_rate)

    reps_df = pd.DataFrame(reps).reset_index(drop=True)
    return reps_df

# ========= 主程序：与需同步版一致地导出两份 CSV =========
def main():
    file_name = "output_druggen_deepseek_V3_MT_with_insight_opt"
    root_dir = f"./{file_name}"

    # 与最新版对齐：默认不删除 inactive；严格筛选；baseline_total=1767
    reps_df = collect_representatives(
        root_dir,
        remove_inactive=False,
        baseline_total=1767,
        strict=True,
    )

    # 计算全数据集层面的统计并导出
    num_df = reps_df.select_dtypes(include="number")
    stats_overall = num_df.agg(["median", "mean", "min", "max"])
    print(stats_overall)
    stats_overall.to_csv(f"{file_name}_report.csv")

    # 导出代表表（包含 PDB_ID、SMILES 以及其他属性）
    cols = []
    if "PDB_ID" in reps_df.columns:
        cols.append("PDB_ID")
    if "SMILES" in reps_df.columns:
        cols.append("SMILES")

    # 其余属性列（保持你原来导出的风格）
    property_cols = [c for c in reps_df.columns if c not in ["PDB_ID", "SMILES"]]
    cols += property_cols

    reps_df.to_csv(f"{file_name}_representative_table.csv", columns=cols, index=False)

if __name__ == "__main__":
    main()