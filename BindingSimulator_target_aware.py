import os
import shutil
import subprocess
import tempfile
import yaml
import json
import sys
import wandb
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from vina import Vina
import requests
sys.stdout.reconfigure(encoding='utf-8')
from pdb import set_trace as st
import time

from meeko import MoleculePreparation
from rdkit import Chem


import subprocess

def convert_pdb_to_pdbqt_with_obabel(pdb_path, pdbqt_path):
    # print("start convert with Open Babel")
    try:
        result = subprocess.run(
            ["obabel", "-ipdb", pdb_path, "-opdbqt", "-O", pdbqt_path, "-xr"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        # print("✅ Open Babel 输出信息：")
        # print(result.stdout)
        # print(result.stderr)

    except subprocess.CalledProcessError as e:
        # print("❌ Open Babel 转换失败（可能是 kekulize 问题）")
        # print("stderr:\n", e.stderr)
        # 返回一个标志值或直接 raise/return None 取决于你想怎么处理
        return False

    # print(f"✅ 转换完成：{pdb_path} -> {pdbqt_path}")
    return True


class BindingSimulator:
    def __init__(self, out_dir="boltz_output", use_msa_server=True, accelerator="gpu", cache_file="binding_cache.json", need_cache=False):
        """
        Initializes the BindingSimulator with local caching.

        Args:
            out_dir (str): Directory to save the predictions.
            use_msa_server (bool): Whether to use the MSA server for multiple sequence alignments.
            accelerator (str): Which accelerator to use: "gpu", "cpu", or "tpu".
            cache_file (str): Path to the local cache file.
            need_cache (bool): Whether to use cache for results.
        """
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)
        self.use_msa_server = use_msa_server
        self.accelerator = accelerator
        self.need_cache = need_cache

        # Load or initialize cache
        self.cache_file = cache_file
        self.cache = self._load_cache() if need_cache else {}

    def _load_cache(self):
        """Loads the cache from a file if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Saves the cache to a file if caching is enabled."""
        if self.need_cache:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=4)

    def safe_rmtree(self, path):
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)

    def _cleanup_output_dir(self):
        """Deletes the Boltz-1 output directory to prevent interference in future runs."""
        if os.path.exists(self.out_dir):
            print(f"🧹 Cleaning up: Removing {self.out_dir} to avoid conflicts in new predictions...")
            self.safe_rmtree(self.out_dir)
        else:
            print(f"✅ No cleanup needed: {self.out_dir} does not exist.")

    def predict_binding_vina(self, protein_seq, drug_smiles, output_format="pdb", box_size=20, exhaustiveness=32, pdb_id="NotAvailable"):
        """
        Predicts the binding confidence scores and structure using either known PDB structure or Boltz-1 prediction.
        Uses cache if enabled.

        Args:
            protein_seq (str): The protein sequence in FASTA format.
            drug_smiles (str): The drug structure in SMILES format.
            output_format (str): Output format for the predictions, either "pdb" or "mmcif".
            pdb_id (str): PDB ID if known, otherwise "NotAvailable".
        """

        self._cleanup_output_dir()
        cache_key = f"{protein_seq}_{drug_smiles}"

        if self.need_cache and cache_key in self.cache:
            return self.cache[cache_key]
        receptor_pdb_path = None
        # 处理已知PDB ID的情况
        if pdb_id != "NotAvailable":
            protein_db_dir = "protein_DB"
            os.makedirs(protein_db_dir, exist_ok=True)
            pdb_file_path = os.path.join(protein_db_dir, f"{pdb_id}.pdb")

            if not os.path.exists(pdb_file_path):
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    with open(pdb_file_path, 'w') as f:
                        f.write(response.text)
                    print(f"已下载原始 PDB 文件：{pdb_file_path}")
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"PDB下载失败 {pdb_id}: {str(e)}")
            receptor_pdb_path = self.out_dir
        # 使用Boltz-1预测结构
        else:
            print("Due to no PDB, use boltz-1 to predict the structure")
            self.out_dir = os.path.join(self.out_dir, "boltz_output")
            with tempfile.TemporaryDirectory() as temp_dir:
                # 创建输入文件
                input_data = {
                    'sequences': [
                        {'protein': {'id': 'A', 'sequence': protein_seq}},
                        {'ligand': {'id': 'L', 'smiles': drug_smiles}}
                    ]
                }
                input_file = os.path.join(temp_dir, "input.yaml")
                with open(input_file, 'w') as f:
                    yaml.dump(input_data, f)

                # 运行Boltz-1预测
                boltz_command = [
                    "boltz", "predict", input_file,
                    "--out_dir", self.out_dir,
                    "--accelerator", self.accelerator,
                    "--output_format", output_format
                ]
                if self.use_msa_server:
                    boltz_command.append("--use_msa_server")

                subprocess.run(boltz_command, check=True)
                receptor_pdb_path = os.path.join(
                    self.out_dir, "boltz_results_input", 
                    "predictions", "input", "input_model_0.pdb"
                )

        try:
        # if True:
            if not os.path.exists(receptor_pdb_path):
                os.makedirs(receptor_pdb_path, exist_ok=True)    

            # 记录分子结构
            # wandb.log({"Binding Structure": wandb.Molecule(open(receptor_pdb_path))})

            # 转换受体格式
            receptor_pdbqt = os.path.join(receptor_pdb_path, f"receptor.pdbqt")
            # print(f"=================={receptor_pdbqt}==================")
            # subprocess.run(
            #     ["obabel", "-ipdb", pdb_file_path, "-opdbqt", 
            #      "-O", receptor_pdbqt, "-xr"],
            #     check=True
            # )
            receptor_pdbqt = os.path.join(receptor_pdb_path, "receptor.pdbqt")
            success = convert_pdb_to_pdbqt_with_obabel(pdb_file_path, receptor_pdbqt)
            if not success:
                print("⚠️ 转换失败，跳过该受体或记录日志")
                return {
                    "error": "Open Babel 转换失败",
                    "binding_affinity": None,
                    "best_pose": None
                }

            # 准备配体
            mol = Chem.MolFromSmiles(drug_smiles)
            if mol is None:
                raise ValueError("无效的SMILES格式")            

            mol = Chem.AddHs(mol)  # ✅ 添加显式氢原子
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
            AllChem.ComputeGasteigerCharges(mol)            

            preparator = MoleculePreparation()
            preparator.prepare(mol)         

            ligand_pdbqt = preparator.write_pdbqt_string()

            # 分子对接
            v = Vina(sf_name='vina', verbosity=1)
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)

            # 计算对接盒中心
            with open(receptor_pdbqt) as f:
                coords = [
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    for line in f if line.startswith("ATOM")
                ]
                center = [sum(c[i] for c in coords)/len(coords) for i in range(3)]

            v.compute_vina_maps(center=center, box_size=[box_size]*3)
            v.dock(
                exhaustiveness=exhaustiveness,
                n_poses=1,
                min_rmsd=1.0,
                max_evals=10_000_000
            )

            # 保存结果
            results = {
                "binding_affinity": v.energies()[0][0],
                # "best_pose": v.poses(),
                # "receptor_source": "PDB" if pdb_id != "NotAvailable" else "Boltz-1"
            }

        except Exception as e:
            print(f"对接失败: {str(e)}")
            results = {
                "error": str(e),
                "binding_affinity": None,
                "best_pose": None
            }

        # 缓存结果
        if self.need_cache:
            self.cache[cache_key] = results

        return results

    def CaculateAffinity(self, protein_seq, drug_smiles, output_format="pdb", box_size=20, exhaustiveness=9, pdb_id="NotAvailable"):
        # self._cleanup_output_dir()
        cache_key = f"{protein_seq}_{drug_smiles}"

        if self.need_cache and cache_key in self.cache:
            return self.cache[cache_key]
        from pdb import set_trace
        

        try:
            # 1. 下载或预测 PDB
            if pdb_id != "NotAvailable":
                pdb_file_path = os.path.join("protein_DB", f"{pdb_id}.pdb")
                os.makedirs("protein_DB", exist_ok=True)
                if not os.path.exists(pdb_file_path):
                    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    with open(pdb_file_path, 'w') as f:
                        f.write(response.text)
                    print(f"已下载 PDB 文件: {pdb_file_path}")
            else:
                raise ValueError("当前仅支持使用已知 PDB ID 模式 (与 AlphaDrug 一致)")
            # set_trace()
            # 2. SMILES 转换为 .pdb 文件
            mol = Chem.MolFromSmiles(drug_smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            mol = Chem.RemoveHs(mol)
            ligand_pdb_path = os.path.join(self.out_dir, f"ligand_{time.time():.0f}.pdb")
            Chem.MolToPDBFile(mol, ligand_pdb_path)

            # 3. 构建 smina 命令
            output_tmp_path = os.path.join(self.out_dir, f"smina_out_{time.time():.0f}")
            smina_cmd = [
                "smina", "-r", pdb_file_path, "-l", ligand_pdb_path,
                "--autobox_ligand", pdb_file_path,
                "--autobox_add", "10",
                "--seed", "1000",
                "--exhaustiveness", str(exhaustiveness),
                ">>", output_tmp_path
            ]
            smina_cmd_str = " ".join(smina_cmd)
            # print(f"[Smina Command] {smina_cmd_str}")

            subprocess.run(smina_cmd_str, shell=True, stdout=subprocess.PIPE)

            # 4. 提取 Affinity 分数
            affinity = 500
            with open(output_tmp_path, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) == 4 and tokens[0] == '1':
                        affinity = float(tokens[1])
                        break

            # 5. 清理临时文件
            subprocess.run(f'rm -rf {output_tmp_path}', shell=True)
            subprocess.run(f'rm -rf {ligand_pdb_path}', shell=True)

            if affinity == 500:
                print("❌ affinity error")
                results = {"binding_affinity": 500.0}
                raise ValueError("Affinity extraction failed, please check the output file.")
            else:
                results = {
                    "binding_affinity": affinity
                }
            print(results)
        except Exception as e:
            print(f"❌ Docking Exception: {e}")
            results = {
                "error": str(e)
            }

        if self.need_cache:
            self.cache[cache_key] = results
        return results


if __name__ == "__main__":
    simulator = BindingSimulator(need_cache=False)  # 关闭缓存用于调试

    protein_sequence = "MKTIIALSYIFCLVFA"  # 示例蛋白质序列
    drug_smiles = "CC(C)Cc1ccccc1"        # 示例 SMILES 分子

    print("🔬 Predicting binding confidence using Vina docking...")
    results = simulator.predict_binding(
        protein_seq=protein_sequence,
        drug_smiles=drug_smiles,
        output_format="pdb",
        box_size=20,
        exhaustiveness=9,
        pdb_id="1zys"  # 使用已知 PDB ID，和你 AlphaDrug 保持一致
    )

    print("📊 Predicted Binding Results:")
    print(json.dumps(results, indent=4, ensure_ascii=False))    


    print("🔬 Predicting binding confidence using smina docking...")
    results = simulator.CaculateAffinity(
        protein_seq=protein_sequence,
        drug_smiles=drug_smiles,
        output_format="pdb",
        box_size=20,
        exhaustiveness=9,
        pdb_id="1zys"  # 使用已知 PDB ID，和你 AlphaDrug 保持一致
    )

    print("📊 Predicted Binding Results:")
    print(json.dumps(results, indent=4, ensure_ascii=False))