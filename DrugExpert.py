import json
import os
import torch
from accelerate.commands.config.config_args import cache_dir
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from rdkit import Chem
import logging
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdmolops import GetMolFrags
from meeko import MoleculePreparation
from utils import calcScore  # 保证你的 utils.py 在 PYTHONPATH 或同目录下
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import logging
import sys
import os
from utils import save_to_llama_format, filter_chinese_characters
sys.stdout.reconfigure(encoding='utf-8')
# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DrugExpert:
    def __init__(self, model_name_or_id="OpenDFM/ChemDFM-v1.5-8B", device="cuda", cache_file="drug_cache.json",
                 data_file="drug_expert_data.json", need_cache=False, cache_dir="./jsons/"):
        """
        Initializes DrugExpert with a pretrained Llama model and an optional local cache.

        Args:
            model_name_or_id (str): Pretrained model name.
            device (str): Device to run the model ("cuda" or "cpu").
            cache_file (str): Path to the local cache file.
            data_file (str): Path to store input/output records.
            need_cache (bool): Whether to use cache for results.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)
        self.model = LlamaForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map=device)
        self.device = device
        self.need_cache = need_cache
        self.data_file = cache_dir+data_file
        self.model_name_or_id = model_name_or_id

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 确保数据文件存在
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump([], f)

        # Load or initialize cache
        self.cache_file = cache_dir+cache_file
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

    def generate_molecule(self, properties_text, do_sample=True, top_k=50, top_p=0.8, temperature=1.0,
                      max_new_tokens=128) -> str:
        """
        Generates a molecule that can pass Meeko preparation and returns a valid SMILES sequence.
        """
        input_text = f"""[Round 0]
        Human: Generate a molecule meeting these requirements: {properties_text}
        Assistant: The generated SMILES is:"""
        instruction = "Generate a molecule with specific properties."

        max_attempts = 50
        attempts = 0
        meeko_failed_list = []

        while attempts < max_attempts:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            generation_config = GenerationConfig(
                do_sample=do_sample, 
                top_k=top_k, 
                top_p=top_p, 
                temperature=temperature,
                max_new_tokens=max_new_tokens, 
                repetition_penalty=1.02, 
                eos_token_id=self.tokenizer.eos_token_id
            )

            outputs = self.model.generate(**inputs, generation_config=generation_config)
            generated_tokens = outputs[:, inputs.input_ids.shape[-1]:]
            generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

            validation_passed = False
            mol = None

            try:
                mol = Chem.MolFromSmiles(generated_text)
                if not mol:
                    raise ValueError("Invalid SMILES syntax")
                if len(GetMolFrags(mol)) > 1:
                    raise ValueError("Multiple fragments detected")
                _, qed, _, mw, sa_score, _ = calcScore(mol)
                if mw > 500:
                    raise ValueError(f"Molecular weight {mw:.1f} out of range")
                # if qed < 0.5:   #ds gpt in CD2020
                # if qed < 0.35:  #ds gpt in druggen
                if qed < 0.4:
                    raise ValueError(f"QED {qed:.2f} too low")
                # if sa_score < 0.5: #ds gpt in CD2020
                if sa_score < 0.4: #ds gpt in druggen
                    raise ValueError(f"SA {sa_score:.2f} too low")
                
                allowed_elements = {6,1,7,8,16,9,17,35,15}
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() not in allowed_elements:
                        raise ValueError(f"Contains forbidden element: {atom.GetSymbol()}")

                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol) == -1:
                    raise RuntimeError("3D conformation generation failed")

                preparator = MoleculePreparation()
                if not preparator.prepare(mol):
                    meeko_failed_list.append(generated_text)
                    raise RuntimeError("Meeko preparation failed")

                pdbqt_str = preparator.write_pdbqt_string()
                if "BRANCH" not in pdbqt_str or "ATOM" not in pdbqt_str:
                    raise ValueError("Invalid PDBQT format")

                validation_passed = True

            except Exception as e:
                logging.warning(f"Validation failed: {str(e)}")
                attempts += 1
                continue

            if validation_passed:
                logging.info(f"✅ Valid SMILES generated: {generated_text}")
                save_to_llama_format(instruction, properties_text, generated_text, self.data_file)
                return generated_text
            # return generated_text
            raise ValueError("❌ Failed to generate valid SMILES.")

        # # 兜底处理：重新尝试之前Meeko失败的SMILES
        # for smi in meeko_failed_list:
        #     try:
        #         mol = Chem.MolFromSmiles(smi)
        #         mol = Chem.AddHs(mol)
        #         if AllChem.EmbedMolecule(mol) == -1:
        #             continue
        #         preparator = MoleculePreparation()
        #         if preparator.prepare(mol):
        #             pdbqt_str = preparator.write_pdbqt_string()
        #             if "BRANCH" in pdbqt_str and "ATOM" in pdbqt_str:
        #                 logging.info(f"♻️ Recovered valid SMILES from Meeko-failed list: {smi}")
        #                 save_to_llama_format(instruction, properties_text, smi, self.data_file)
        #                 return smi
        #     except Exception as e:
        #         logging.warning(f"Retry Meeko failed: {str(e)}")
        #         continue

        # raise ValueError("❌ Failed to generate valid SMILES after multiple attempts and Meeko retry.")

    def predict_toxicity_and_promiscuity(self, smiles: str, max_new_tokens=128) -> dict:
        """
        Predicts the toxicity and promiscuity of a given molecule.

        Args:
            smiles (str): The SMILES representation of the molecule.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            dict: A **comprehensive analysis**, including toxicity, promiscuity, and metabolism.
        """
        instruction = "Analyze the toxicity and promiscuity of a given molecule."

        toxicity_report = {}

        for round_num, prompt in enumerate([
            "Analyze the molecular structure and identify functional groups associated with toxicity.",
            "Predict how this molecule is likely to be metabolized in the human body.",
            "Check if this molecule shares structural similarity with known toxic compounds."
        ], start=1):
            input_text = f"[Round {round_num}]\nHuman: {prompt}\nSMILES: {smiles}\nAssistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            response_text = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]

            toxicity_report[prompt] = response_text

            # 存储输入/输出
            save_to_llama_format(instruction, f"SMILES: {smiles}, Task: {prompt}", response_text, self.data_file)

        return toxicity_report

    def chat(self, text_input: str, instruction="Engage in the inference about drug discovery and molecular design.", max_new_tokens=512) -> str:
        input_text = f"Human: {text_input} Assistant: "
        generation_config = GenerationConfig(
            do_sample=True, top_k=50, top_p=0.9, temperature=1.0,
            max_new_tokens=max_new_tokens, repetition_penalty=1.2, eos_token_id=self.tokenizer.eos_token_id
        )
        inputs = self.tokenizer(instruction+" "+input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        response_text = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        save_to_llama_format(instruction, text_input, response_text, self.data_file)
        return filter_chinese_characters(response_text.strip())


    def compute_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            MolLogP, qed, tpsa, MolWt, sa_score, np_score = calcScore(mol)
        except Exception as e:
            logging.warning(f"calcScore failed: {str(e)}")
            return None

        props = {
            "OctanolWaterPartitionCoefficientS(↑)": MolLogP,
            "QuantitativeEstimateOfDruglikeness(↑)": qed,
            "TopologicalPolarSurfaceArea": tpsa,
            "MolecularWeight": MolWt,
            "NormalizedSyntheticAccessibilityScore(↑)": sa_score,
            "NaturalProductLikenessScore(↑)": np_score,
            # 可选补充字段
            # "ExactMolWt": Descriptors.ExactMolWt(mol),
            # "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol),
            # "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
            # "NumAtoms": mol.GetNumAtoms(),
            # "NumBonds": mol.GetNumBonds(),
            # "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
            # "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
            # "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
            # "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
            # "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            # "FractionCSP3": Descriptors.FractionCSP3(mol),
            # "LabuteASA": rdMolDescriptors.CalcLabuteASA(mol),
        }

        try:
            Chem.Kekulize(mol)
            props["MolMR"] = Descriptors.MolMR(mol)
        except Exception as e:
            logging.warning(f"Kekulization failed: {str(e)}")
            props["MolMR"] = None

        # 可选：Chi/Kappa 指数
        # for name in ["Chi0", "Chi1", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "Kappa1", "Kappa2", "Kappa3"]:
        #     try:
        #         props[name] = getattr(Descriptors, name)(mol)
        #     except Exception:
        #         props[name] = None

        return props

# Example usage
if __name__ == "__main__":

    drug_expert = DrugExpert(need_cache=False)  # Enable cache

    # # Define molecular properties
    # properties = {
    #     "logP": 2.5,
    #     "Molecular weight": 300,
    #     "Toxicity": "Low"
    # }

    # print("🧪 Generating candidate molecule...")
    # generated_smiles = drug_expert.generate_molecule(f"{properties}")
    # print(f"🔬 Generated SMILES: {generated_smiles}")

    # print("⚠️ Predicting toxicity & promiscuity...")
    # toxicity_results = drug_expert.predict_toxicity_and_promiscuity(generated_smiles)
    # print(f"📊 Toxicity & Promiscuity Prediction: {toxicity_results}")

    # molecular_property = drug_expert.compute_properties(generated_smiles)
    # print(f"Properties:{molecular_property}")
    smiles = "Cc1ccc(-c2ccc(NC(=O)/C=C/CNC(=O)c3ccc(NC(=O)c4ccccc4)cc3)cc2)nc1Cl"
    from pdb import set_trace
    set_trace()
    for i in range(5):
        question = input("Please input your question:")
        response = drug_expert.chat(question)
        print("Expert:",response)