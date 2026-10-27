# CadreAI

**CadreAI: Agentic AI for Semantically Decoupled Protein–Drug Design and Optimization**

CadreAI is an agentic framework for protein-guided molecular generation and iterative optimization. This repository provides scripts for the CrossDocked2020 and DrugGen experiments, candidate evaluation, and reproducibility materials.

## ⚙️ Installation

We recommend Conda and Python 3.10. The following commands reproduce the **dependency installation procedure** used for the project; they are not a version-locked environment. For exact reproduction, archive a tested `environment.yml` and record the CUDA driver, package versions, model revisions, and executable versions from the experimental machine.

```bash
conda create -n cadreai python=3.10 -y
conda activate cadreai

# Install the PyTorch build compatible with your CUDA driver.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers huggingface_hub accelerate 'accelerate>=0.26.0'
pip install --upgrade boltz
pip install gradio_client rdkit vina meeko biopython openai
pip install seaborn ollama loguru

conda install -c conda-forge openbabel -y
conda install -c bioconda smina -y

# Authenticate if the required model checkpoints are gated.
hf auth login
```

> **Version pinning:** The commands above specify packages, not the exact versions used for the reported experiments. Before release, export the validated environment (`conda env export --no-builds > environment.yml`) and commit it. Do not treat an environment exported on a different machine as verified.

## 📊 Experiments

The benchmark target/PDB ID lists are provided in `datasets/`.

### CrossDocked2020

Download the structures using `datasets/CrossDocked2020/test/get_list.py`, then launch the experiment:

```bash
python datasets/CrossDocked2020/test/get_list.py
bash run_CD.sh
```

For each of 100 targets, CadreAI uses **25 generated initial candidates** followed by **five optimization rounds of 15 candidates each**, for **100 accepted candidates per target**. The reference ligand is **not** included among the initial candidates. Raw generation attempts, invalid molecules, and duplicates are not equivalent to accepted candidates.

### DrugGen

The experiment script downloads the required PDB files automatically:

```bash
bash run_DG.sh
```

The nominal CadreAI budget is **3 initial candidates plus 5 rounds × 3 variants = 18 candidates per target**. The five-method comparison in the revised study uses **802 targets** with valid evaluations for every compared method. DrugGen docking uses the specified receptor preparation and chain A for all five methods.

## 🔍 Evaluation

Set the experiment output directory in `evaluate.py`, then run:

```bash
python evaluate.py
```

The evaluation selects one representative molecule **per target** from the valid evaluated candidates, using either composite-score ranking or docking-score ranking. **Do not filter candidates by the hit thresholds before selecting Top-1.** Hit rate is calculated *after* representative selection.

The implemented composite score is:

```text
composite = 0.5 * QED - 0.5 * Vina + 0.4 * SA_norm
SA_norm = clip((10 - SA_raw) / 9, 0, 1)
```

The hit criteria are `Vina < -8.18 kcal/mol`, `QED > 0.25`, and `SA_norm > 0.59` (equivalently `SA_raw < 4.69`, approximately). The composite uses raw docking values, not min–max-normalized docking scores.

For CrossDocked2020, report target-level statistics over the **100 target representatives** and hit rate with a denominator of **100 targets**. Report targets without a valid evaluated candidate separately; do not silently remove them from the hit-rate denominator. For DrugGen, report valid-evaluation coverage against **1,767 total targets**, and use the **802-target intersection** for the five-method matched comparison. Exclude failed docking calculations and sentinel values from numerical docking summaries consistently across methods; report failure counts separately.

## 🧪 Reproducibility and Experimental Settings

| Item | Setting |
| --- | --- |
| CrossDocked2020 budget | 25 initial + 5 × 15 = 100 accepted candidates per target |
| DrugGen budget | 3 initial + 5 × 3 = 18 nominal candidates per target |
| Composite weights (QED, −Vina, SA) | 0.5, 0.5, 0.4 |
| CrossDocked2020 docking | `smina`, ligand-centered autobox, `autobox_add=6 Å`, `exhaustiveness=9`, `seed=1234` |
| CrossDocked2020 reference ligand | Used to define the docking box, **not** as an initial generated candidate |
| DrugGen docking | Receptor-centered box, chain A receptor |
| LLM for revised benchmark experiments | DeepSeek Flash |
| SGIC insight management | String-based title merging and entry deduplication using Python `difflib` |

The SGIC implementation uses text similarity rather than embedding-based clustering or a minimum-support/consistency gate. The final released code and experimental settings should be treated as authoritative for exact thresholds and runtime configuration.

**To complete reproducibility before publication:** archive the exact planner and generator decoding parameters, model identifiers and access dates, molecular preparation/protonation steps, docking output modes, DrugGen target-to-structure mapping, Boltz-2 version and MSA/sampling settings, and number of independent runs. These details should match the revised manuscript and Supporting Information; they are not inferred from the installation commands.

### Reproducibility assets

The repository release should include or link to the following artifacts, with their actual paths documented here once uploaded:

- Cached LLM request–response JSON records (including inputs and outputs), for replaying recorded model responses without dependence on future hosted model versions.
- Cached protein-analysis reports to reduce dependence on the external protein-analysis service.
- Full per-round optimization trajectories and extracted Insight Libraries.
- Docked molecular poses and baseline candidate sets used for comparisons.
- Benchmark target lists, evaluation outputs, and per-target failure/coverage records.

**Release status:** The items above are a publication checklist, **not a claim that all files are currently available**. Add links to the actual repository files or archives after uploading them. Cached model outputs enable deterministic inspection/replay of those outputs, but do not guarantee identical fresh LLM responses or identical downstream docking results on different software builds.

## 📚 Citation

If you use CadreAI in your research, please cite the manuscript (currently under review):

> Pan, F., Lyu, F., and Yuen, P. C. *CadreAI: Agentic AI for Semantically Decoupled Protein–Drug Design and Optimization*. Journal of Chemical Information and Modeling, under review (2026).

A formal BibTeX entry should be added after publication.

## 📜 License

MIT License. See [LICENSE](LICENSE) for details. **Include the `LICENSE` file in the repository before publishing this statement.**
