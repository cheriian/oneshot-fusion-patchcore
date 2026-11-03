# One-Shot PatchCore with a WRN-50 + DINOv2 Fusion Backbone

CS3807 Deep Learning Laboratory mini-project, Shiv Nadar University Chennai.

Extends **PatchCore** (Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022)
in the **one-shot** setting (one nominal training image per category) on **MVTec AD**, using the paper's
**PatchCore-10** configuration (10% greedy coreset, 224×224 input). All models use the same training image
for each of 10 seeds × 15 categories.

| Model | Backbone |
|---|---|
| PatchCore (baseline, paper) | WideResNet-50, layers 2+3 |
| DINOv2 PatchCore | DINOv2 ViT-S/14, blocks 7+11, native 16×16 grid |
| **Fusion PatchCore (ours)** | WRN-50 and DINOv2 patch features, L2-normalised and concatenated on a 28×28 grid |

## Results (1-shot MVTec AD, mean ± std over 10 seeds)

| Method | Image AUROC (%) | Pixel AUROC (%) |
|---|---|---|
| Paper, PatchCore-10 (Table S5) | 83.4 ± 0.6 | 92.0 ± 0.2 |
| Paper, PatchCore-25 (Table S5, reference) | 84.1 ± 0.7 | 92.4 ± 0.3 |
| Ours, WRN-50 (reproduction) | 83.29 ± 1.42 | 91.89 ± 0.32 |
| Ours, DINOv2 | 85.62 ± 0.83 | 91.93 ± 0.57 |
| **Ours, Fusion** | **88.05 ± 0.87** | **94.50 ± 0.24** |

- The reproduction matches the paper (no significant difference).
- Fusion beats both single backbones on every seed (Wilcoxon p = 0.002, both metrics).
- Fusion removes 28% of the paper's image-level error and 31% of its pixel-level error.
- All our runs use the PatchCore-10 setting; Fusion also exceeds the paper's PatchCore-25 one-shot result (+3.95 image, +2.10 pixel AUROC).

### Ablation

| Variant | Image AUROC (%) | Pixel AUROC (%) |
|---|---|---|
| WRN-50 only, L2-normalised (α = 1) | 81.95 ± 1.16 | 91.50 ± 0.31 |
| DINOv2 only, aligned to 28×28 grid (α = 0) | 88.08 ± 0.88 | 93.80 ± 0.29 |
| Fusion (α = 0.5) | 88.05 ± 0.87 | **94.50 ± 0.24** |

- Aligning DINOv2 to PatchCore's 28×28 patch grid drives the detection gain.
- Adding WRN-50 does not change detection, but improves localisation on all 10 seeds (+0.69 pixel AUROC, p = 0.002).

## Repository contents
- `oneshot_patchcore.py`: experiment script (`--backbone wrn50 | dinov2 | fusion`, `--alpha`); resumable, one CSV row per category and seed
- `analyze_results.py`: four-way comparison, paired Wilcoxon tests, ablation, per-category tables and report figures
- `results/results.csv`: raw results of every run
- `results/report_assets/`: tables and figures used in the report
- `notebook/patchcore_oneshot.ipynb`: the Colab notebook that produced the results (setup, experiments, ablation, analysis)

## Reproduce (Google Colab, T4 GPU)
1. Runtime → Change runtime type → **T4 GPU**.
2. Download MVTec AD from the [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to `/content/mvtec`.
3. Clone the official PatchCore code and install dependencies:
   ```
   git clone https://github.com/amazon-research/patchcore-inspection.git
   cd patchcore-inspection && pip install faiss-cpu pretrainedmodels timm
   ```
4. Copy `oneshot_patchcore.py` into `patchcore-inspection/` and run (10 seeds by default):
   ```
   PYTHONPATH=src python oneshot_patchcore.py --data /content/mvtec --backbone wrn50  --out results.csv
   PYTHONPATH=src python oneshot_patchcore.py --data /content/mvtec --backbone dinov2 --out results.csv
   PYTHONPATH=src python oneshot_patchcore.py --data /content/mvtec --backbone fusion --out results.csv
   PYTHONPATH=src python oneshot_patchcore.py --data /content/mvtec --backbone fusion --alpha 1.0 --out results.csv
   PYTHONPATH=src python oneshot_patchcore.py --data /content/mvtec --backbone fusion --alpha 0.0 --out results.csv
   ```
5. Analyse: `RESULTS_DIR=<folder containing results.csv> python analyze_results.py`

## Acknowledgements
Built on the official [PatchCore implementation](https://github.com/amazon-research/patchcore-inspection) (Amazon)
and [DINOv2](https://github.com/facebookresearch/dinov2) (Meta). The MVTec AD dataset is not redistributed here;
see its official page for licence terms.