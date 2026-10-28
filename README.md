# CadreAI

**CadreAI: Agentic AI for Semantically Decoupled Protein–Drug Design and Optimization**

CadreAI is an agentic framework for protein-guided molecular generation and iterative optimization. This repository provides experiment scripts for CrossDocked2020 and DrugGen, molecular evaluation utilities, and supporting reproducibility assets.

## ⚙️ Installation

We recommend Conda and Python 3.10. The commands below describe a dependency installation procedure, **not** a fully version-locked reproduction environment. Select a PyTorch build compatible with your CUDA driver.

```bash
conda create -n cadreai python=3.10 -y
conda activate cadreai

# Example CUDA 12.6 PyTorch wheels; change as appropriate for your system.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install transformers huggingface_hub 'accelerate>=0.26.0'
pip install --upgrade boltz
pip install gradio_client rdkit vina meeko biopython openai
pip install seaborn ollama loguru tqdm

conda install -c conda-forge openbabel -y
conda install -c bioconda smina -y

# Only if a required checkpoint is gated:
hf auth login
```

For exact reproducibility, record the tested Python and package versions, CUDA driver, model checkpoint revisions, and external executable versions. An environment export from a different machine should not be treated as experimentally validated.

## 📊 Experiments

Benchmark target/PDB ID lists are maintained under `datasets/`.

### CrossDocked2020

Download the benchmark structures and launch the experiment:

```bash
python datasets/CrossDocked2020/test/get_list.py
bash run_CD.sh
```

The experimental design allocates **25 generated initial candidate slots** and **five optimization rounds of 15 candidate slots each**, totaling **100 candidate slots per target** across 100 protein targets. The benchmark reference ligand defines the docking search region; **it is not an initial candidate**. Generation attempts, accepted candidate slots, distinct canonical SMILES, and successfully docked molecules are different quantities and should be reported accordingly.

### DrugGen

The experiment script retrieves the required PDB structures:

```bash
bash run_DG.sh
```

The nominal CadreAI budget is **3 initial candidates + 5 rounds × 3 variants = 18 candidates per target**. The revised five-method comparison uses **802 common successfully evaluated protein targets**. The common evaluation protocol uses cleaned chain A receptors across all five methods.

## 🔍 CrossDocked2020 Evaluation

The revised `evaluate.py` implements **unconditional target-level Top-1 selection**. It reads all three molecule archives for each target, canonicalizes SMILES, merges duplicate records, and selects **two independent representatives** from the same evaluable candidate pool:

- **Composite Top-1:** maximum composite score.
- **Docking Top-1:** minimum Vina docking score.

**Hit thresholds are not applied before Top-1 selection.** The hit flag is calculated separately for each selected representative. Hit rate (HR) always uses **100 protein targets** as its denominator, including targets without evaluable molecules.

Run from the repository root:

```bash
python evaluate.py --root CadreAI_Coupled --output-dir evaluation
```

Replace `CadreAI_Coupled` with the actual experiment output directory. The evaluator reads the following sources for each numeric target ID `0`–`99`:

```text
<root>/<target_id>/molecules.csv
<root>/cache/<target_id>/molecules.csv
<root>/cache/<target_id>/molecules_pool.csv
```

The input files must contain `SMILES`, `vina`, `qed`, `sa`, and `composite` for evaluated molecules. Records without complete metrics cannot be selected. The evaluator writes:

| Output | Description |
| --- | --- |
| `evaluation/CrossDocked2020_composite_top1.csv` | Composite-ranked Top-1 SMILES and metrics for each evaluable target |
| `evaluation/CrossDocked2020_docking_top1.csv` | Docking-ranked Top-1 SMILES and metrics for each evaluable target |
| `evaluation/CrossDocked2020_target_coverage.csv` | Per-target candidate counts, source coverage, and duplicate diagnostics |
| `evaluation/CrossDocked2020_evaluation_summary.json` | Mean, median, SD, HR, missing targets, and evaluation warnings |

If any target lacks evaluable molecules, the script writes provisional outputs and exits with a nonzero status unless `--allow-incomplete` is explicitly supplied. Missing targets are **never silently removed from the HR denominator**. Numerical means and medians summarize selected evaluable representatives; coverage and failures are reported separately.

For additional validation:

```bash
python evaluate.py \
  --root CadreAI_Coupled \
  --output-dir evaluation \
  --strict-duplicates \
  --strict-composite
```

The default duplicate policy follows the accompanying `monitor.py`: canonical-SMILES duplicates retain the record with the lowest Vina, and conflicting scores are flagged. `--strict-duplicates` rejects conflicting duplicate metrics instead. Use a consistent, documented duplicate policy when comparing methods.

The composite and normalized synthetic-accessibility scores are defined as:

```text
composite = 0.5 * QED - 0.5 * Vina + 0.4 * SA_norm
SA_norm = clip((10 - SA_raw) / 9, 0, 1)
```

The hit criteria are **Vina < −8.18 kcal/mol**, **QED > 0.25**, and **SA_norm > 0.59** (equivalently, SA_raw < 4.69). Composite ranking uses raw docking values, not min–max-normalized docking scores. The evaluator uses the recorded `composite` values and can check their agreement with this formula using `--strict-composite`.

### DrugGen evaluation reporting

Report valid-evaluation coverage against **1,767 total targets**. The five-method matched comparison uses the **802-target intersection**, while paired method comparisons use their corresponding pairwise target intersections. Failed docking calculations and sentinel values are excluded from numerical docking summaries consistently across methods, with failure counts reported separately. The CrossDocked2020 `evaluate.py` is **not** a DrugGen five-method analysis script.

## 🧪 Reproducibility and Experimental Settings

| Item | Setting |
| --- | --- |
| CrossDocked2020 design budget | 25 initial + 5 × 15 = 100 candidate slots per target |
| DrugGen design budget | 3 initial + 5 × 3 = 18 nominal candidates per target |
| Composite weights (QED, −Vina, SA_norm) | 0.5, 0.5, 0.4 |
| CrossDocked2020 docking | `smina`; reference-ligand-defined autobox; `autobox_add=6 Å`; `exhaustiveness=9`; `seed=1234` |
| CrossDocked2020 reference ligand | Defines docking region only; not used as an initial candidate |
| DrugGen docking | `smina`; cleaned-receptor-defined autobox; `autobox_add=20 Å`; `exhaustiveness=40`; nine requested modes; `seed=1000` |
| DrugGen receptor preparation | Cleaned chain A receptor used consistently across compared methods |
| LLM for revised benchmark experiments | DeepSeek Flash |
| SGIC insight management | String-based title merging and entry deduplication using Python `difflib` |

SGIC uses text similarity rather than embedding-based clustering or a minimum-support/consistency gate. For complete reproducibility, the released source, model identifiers, decoding parameters, docking preparation details, and Supporting Information must be mutually consistent.

### Reproducibility assets

The following are relevant to reproducing the reported experiments. **Their inclusion below is a release checklist, not confirmation that every asset is already uploaded.** Link each item to its actual repository path or an archival download when available.

- Cached LLM request–response JSON records, including inputs and outputs.
- Cached protein-analysis reports.
- Per-round optimization records and Insight Libraries.
- Generated molecules and, where available, docked poses and baseline candidate sets.
- Benchmark target lists, target-to-structure mappings, evaluation outputs, and per-target coverage/failure records.
- Tested environment specifications, exact model identifiers and access dates, decoding settings, molecular preparation settings, and independent-run counts.

Cached LLM outputs support inspection or replay of the **recorded responses**; they do not guarantee identical results from fresh hosted-model calls or docking across different software environments.

## 📚 Citation

If you use CadreAI, please cite the manuscript (currently under review):

> Pan, F., Lyu, F., and Yuen, P. C. *CadreAI: Agentic AI for Semantically Decoupled Protein–Drug Design and Optimization*. Journal of Chemical Information and Modeling, under review (2026).

A formal BibTeX entry will be added upon publication.

## 📜 License

MIT License. See [LICENSE](LICENSE). Ensure that the `LICENSE` file is included in the repository before publishing this statement.
