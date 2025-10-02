import os
import csv
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

from CadreAI import CadreAI
from DrugExpert import DrugExpert
from utils import get_protein_sequence_from_pdb
from Bio.PDB import PDBParser, PPBuilder

# 错误日志写入锁
default_error_lock = threading.Lock()


def parse_arguments():
    parser = argparse.ArgumentParser(description="CadreAI Experiment on DrugGen dataset")
    parser.add_argument("--commander_model", type=str, default="deepseek-V3",
                        help="The Commander model to use for CadreAI")
    parser.add_argument("--round", type=str, default="1",
                        help="The Commander model to use for CadreAI")
    
    return parser.parse_args()


def extract_pdb_title(pdb_path):
    title_lines = []
    with open(pdb_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("TITLE"):
                title_lines.append(line[10:].rstrip())
            if line.startswith("ATOM"):
                break
    return " ".join(title_lines)


def process_pdb(row, row_num, input_csv, output_dir, cache_dir, locks_dir,
                api_key, commander_model, drug_expert, error_log_path):
    # 行校验
    if len(row) < 1:
        print(f"Row {row_num}: 无效行，已跳过")
        return

    pdb_id = row[0].strip()
    os.makedirs(locks_dir, exist_ok=True)
    lock_file = os.path.join(locks_dir, f"{pdb_id}.lock")
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        print(f"{pdb_id} 已被其他线程或程序锁定，跳过")
        return

    pdb_dir = os.path.dirname(input_csv)
    pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    protein_sequence = get_protein_sequence_from_pdb(pdb_id, pdb_dir=pdb_dir)
    protein_title = extract_pdb_title(pdb_file)

    pdb_output_dir = os.path.join(output_dir, pdb_id)
    molecules_file = os.path.join(pdb_output_dir, "molecules.csv")
    smiles_file = os.path.join(pdb_output_dir, "final_molecule.smiles")

    # 跳过已完成
    if all(os.path.exists(f) for f in [molecules_file, smiles_file]):
        print(f"跳过 {pdb_id}：输出文件已存在且有效")
        return

    os.makedirs(pdb_output_dir, exist_ok=True)

    try:
        print(f"\n正在处理 {pdb_id}: {protein_title} ({row_num-1})")
        drug_voyager = CadreAI(
            api_key=api_key,
            commander_model=commander_model,
            drug_model_itself=drug_expert,
            max_iterations=10,
            cache_dir=cache_dir,
            description=f"{pdb_id}"
        )
        result = drug_voyager.run_drug_discovery_cycle(protein_sequence, protein_title, pdb_id=pdb_id)

        with open(smiles_file, 'w', encoding='utf-8') as mol_file:
            mol_file.write(result.get("final_smiles", ""))

        with open(molecules_file, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["SMILES", "vina", "qed", "sa", "composite"])
            for mol in result.get("molecule_list", []):
                writer.writerow(mol)

    except Exception as e:
        print(f"!! 处理 {pdb_id} 时出错: {str(e)}")
        with default_error_lock:
            with open(error_log_path, 'a', encoding='utf-8') as error_log:
                error_log.write(f"{pdb_id}\t{str(e)}\n")
        # + 删除对应的锁文件
        try:
            os.remove(lock_file)
        except Exception as clean_lock_error:
            print(f"清理失败 锁文件 {lock_file}: {clean_lock_error}")
        


def main(commander_model, api_key, input_csv, output_dir, cache_dir, error_log_path, drug_expert):
    # 读取输入 CSV
    with open(input_csv, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        rows = list(reader)

    # 随机打乱 PDB 列表
    random.shuffle(rows)

    # 创建锁目录
    locks_dir = os.path.join(output_dir, "locks")
    os.makedirs(locks_dir, exist_ok=True)

    # 多线程执行
    num_workers = 5
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for row_num, row in enumerate(rows, start=2):
            futures.append(executor.submit(
                process_pdb,
                row, row_num,
                input_csv, output_dir, cache_dir, locks_dir,
                api_key, commander_model, drug_expert, error_log_path
            ))
        # 等待所有完成
        for f in futures:
            f.result()

    print("\n所有任务处理完成！")


if __name__ == "__main__":
    args = parse_arguments()
    commander_model = args.commander_model
    RD = args.round
    api_key = "EMPTY"
    input_csv = "./datasets/CrossDocked2020/test/pdb_ids.csv"
    output_dir = f"output_CD_{commander_model.replace(':', '_').replace('-', '_')}_MT_with_insight_opt_RD{RD}"
    cache_dir = os.path.join(output_dir, "cache")
    error_log_path = os.path.join(output_dir, "error_log.txt")

    drug_expert = DrugExpert(
        model_name_or_id="OpenDFM/ChemDFM-v1.5-8B",
        cache_file="drug_cache.json",
        need_cache=True,
        device="cuda",
        cache_dir=cache_dir
    )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    main(commander_model, api_key, input_csv, output_dir, cache_dir, error_log_path, drug_expert)
