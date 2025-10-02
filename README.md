# CadreAI

## ⚙️ Installation

We recommend using **conda** to manage dependencies.

```bash
# Create a new environment
conda create -n cadreai python=3.10 -y
conda activate cadreai

# Core dependencies
pip install transformers huggingface
pip install boltz -U
pip install gradio_client
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install rdkit
pip install accelerate>=0.26.0
pip install --upgrade huggingface_hub
pip install vina meeko
pip install biopython
pip install openai

# Additional chemistry tools
conda install -c conda-forge openbabel -y
conda install -c bioconda smina -y

# Authenticate HuggingFace (for model checkpoints)
huggingface-cli login
```


## 📊 Experiment Tracking
We integrate [Weights & Biases](https://wandb.ai/) for logging:  

```bash
wandb login
```

Each experiment run automatically tracks:
- Docking scores (Vina/smina)  
- Drug-likeness (QED)  
- Synthetic accessibility (SA)  
- Success rate and optimization curves  

---

## 📚 Citation
If you use CadreAI in your research, please cite:

```
Pan, F. et al.  
CadreAI: Agentic AI for Mechanism-Aware Protein–Drug Generation and Optimization  
(2025, under review)
```

---

## 📜 License
MIT License.  
See [LICENSE](LICENSE) for details.
