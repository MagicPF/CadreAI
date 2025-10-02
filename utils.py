import logging

from rdkit import Chem
from rdkit.Chem import Draw
import re
import json
import os
import torch
import time
import requests
import subprocess
from Bio.PDB import PDBParser, PPBuilder

def generate_molecule_image(smiles):
    """ Generates an RDKit image for a given SMILES. """
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol) if mol else None

def save_to_llama_format(instruction, input_text, output_text, data_file):
    if "Error" in input_text + output_text or "Failed" in input_text + output_text or "error" in input_text + output_text:
        return
    """
    Saves the input-output pair in LLaMA training format to a JSON file.

    Args:
        instruction (str): The task instruction.
        input_text (str): The user's input.
        output_text (str): The model's response.
    """
    new_entry = {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

    # 读取已有数据
    try:
        with open(data_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # logging.warning("JSON 文件格式错误，重置为空列表")
        data = []

    # 追加新数据
    data.append(new_entry)

    # 写回文件
    with open(data_file, "w") as f:
        json.dump(data, f, indent=4)

    # logging.info(f"Saved interaction to {data_file}")

import re
def filter_chinese_characters(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return pattern.sub('', text)

import json
import os
import datetime

import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    def __init__(self, log_dir="./logs/"):
        """ 初始化日志存储目录 """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在

    def log(self, agent_name, input_data, output_data):
        """ 记录 Agent 的输入和输出 """
        log_file = os.path.join(self.log_dir, f"{agent_name}.log")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_entry = {
            "timestamp": timestamp,
            "input": input_data,
            "output": output_data
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, indent=4, ensure_ascii=False) + "\n\n")


import os, sys, pickle, gzip, json
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
import sascorer
import npscorer
from loguru import logger

# logger.info(Descriptors._descList)
# des_name = [name[0] for name in Descriptors._descList]
# logger.info(des_name)
# fscore = npscorer.readNPModel()
fscore = pickle.load(gzip.open(os.path.join(RDConfig.RDContribDir, 'NP_Score') + '/publicnp.model.gz'))


def calcScore(mol):
    """
        return MolLogP, qed, sa_score, np_score, docking_score
    """
    des_list = ['MolLogP', 'qed', 'TPSA', 'MolWt']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    
    MolLogP, qed, tpsa, MolWt = calculator.CalcDescriptors(mol)
    sa_score = sascorer.calculateScore(mol)
    sa_score = round((10 - sa_score) / 9, 2)
    np_score = npscorer.scoreMol(mol, fscore)
    # docking_score = CaculateAffinity(Chem.MolToSmiles(mol))
    # return MolLogP, qed, sa_score, np_score, docking_score
    return MolLogP, qed, tpsa, MolWt, sa_score, np_score, 

def visDensity(path_list, out_path):
    dataArr = []
    label = ['LogP', 'QED', 'SA', 'NP', 'MolWt', 'Docking']
    color = ['#DA4453', '#4A89DC', '#967ADC', '#D770AD', '#37BC9B']
    legend = ['ligann', 'ours']
    for data_path in path_list:
        with open(data_path, 'r') as f:
            s = json.load(f)
        data = []
        score = s['score']
        smiles = s['validSmiles']

        dict = {}
        no_repeat_smiles = []
        for i, smi in enumerate(smiles):
            if smi in dict:
                dict[smi].append(float(score[i]))
            else:
                dict[smi] = [float(score[i])]
                no_repeat_smiles.append(smi)
        
        for smi, score_arr in tqdm(dict.items()):
            mol = Chem.MolFromSmiles(smi)
            MolLogP, qed, tpsa, MolWt, sa_score, np_score = calcScore(mol)
            data.append([MolLogP, qed, sa_score, np_score, MolWt, np.mean(score_arr)]) 
            # data.append([MolLogP, qed, tpsa, sa_score, np_score]) 
        data = np.array(data)
        logger.info(np.mean(data, axis=0))
        logger.info(np.var(data, axis=0))
        indices = np.random.choice(a=len(data), size=len(data), replace=False, p=None)
        logger.info(calcTanimotoSimilarity([no_repeat_smiles]))
        dataArr.append(data[indices][:,:].T)
    dataArr = np.array(dataArr).transpose(1, 0, 2)
    # print(dataArr.shape)
    plt.figure(figsize=(20, 12))
    
    for i in range(dataArr.shape[0]):
        plt.subplot(231 + i)
        for j in range(dataArr.shape[1]):
            ax = sn.kdeplot(dataArr[i, j, :],color=color[j],shade=True)
        plt.xlabel(label[i], fontsize=18)
        plt.ylabel(' ')
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.legend(legend)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
        
    # data = [calcScore(Chem.MolFromSmiles(smi))[0] for smi in s['validSmiles']]
    # res = sn.kdeplot(data,color='green',shade=True, x="total_bill")

def calcTanimotoSimilarityPairs(s1, s2):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s1), 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s2), 2, nBits=1024)
    return DataStructs.FingerprintSimilarity(fp1,fp2)



def calcTanimotoSimilarity(smiles_arr):
    fpsLlist = []
    for smiles in smiles_arr:
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=1024) for smile in smiles]
        fpsLlist.append(fps)
    
    data = []
    for fps1 in tqdm(fpsLlist):
        for fps2 in fpsLlist:
            IntDiv = []
            for fp1 in fps1:
                for fp2 in fps2:
                    IntDiv.append(DataStructs.FingerprintSimilarity(fp1,fp2))
            data.append(1- np.sqrt(np.sum(IntDiv)/(len(fps1)*len(fps2))))
    
    return data

def CaculateAffinity(smi, file_protein, out_path='./', prefix=''):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid SMILES")
        m2 = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m2)
        m3 = Chem.RemoveHs(m2)
        file_output = os.path.join(out_path, prefix + str(time.time()) + '.pdb')
        Chem.MolToPDBFile(m3, file_output)

        smina_cmd_output = os.path.join(out_path, prefix + str(time.time()))
        launch_args = [
            "smina", "-r", file_protein, "-l", file_output,
            "--autobox_ligand", file_protein,
            "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9",
            ">>", smina_cmd_output
        ]
        subprocess.run(' '.join(launch_args), shell=True, stdout=subprocess.PIPE)

        affinity = 500
        with open(smina_cmd_output, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == '1':
                    affinity = float(parts[1])
                    break

        subprocess.run(f'rm -rf {smina_cmd_output}', shell=True)
        subprocess.run(f'rm -rf {file_output}', shell=True)

    except Exception as e:
        print(f"❌ Affinity error: {e}")
        affinity = 500
    return affinity

def download_pdb_file(pdb_id, pdb_dir="./datasets/druggen"):
    """
    直接下载指定 pdb_id 的原始 PDB 文件，保存到 pdb_dir
    """
    pdb_file_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_dir):
        os.makedirs(pdb_dir, exist_ok=True)
    if not os.path.exists(pdb_file_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(pdb_file_path, 'w', encoding='utf-8') as file:
                file.write(response.text)
            print(f"Downloaded PDB file for {pdb_id}")
        else:
            raise Exception(f"Failed to download PDB file for {pdb_id}, status code {response.status_code}")
    return pdb_file_path

def get_protein_sequence_from_pdb(pdb_id, pdb_dir="./datasets/druggen"):
    """
    提取 PDB 文件中的主链氨基酸序列（只返回第一个链）
    """
    pdb_file_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_file_path):
        print(f"本地不存在 {pdb_id} 的 PDB 文件，开始下载...")
        download_pdb_file(pdb_id, pdb_dir=pdb_dir)

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure(pdb_id, pdb_file_path)

    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            peptides = ppb.build_peptides(chain)
            if peptides:
                sequence = ''.join(str(pp.get_sequence()) for pp in peptides)
                return sequence
    return ""