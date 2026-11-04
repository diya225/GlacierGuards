# Training the segmentation model on Kaggle

Dataset: `glacierguards_kaggle.zip` (built by `scripts/build_kaggle_dataset.py`): lake-year tiles (`tiles/*.npz`), `splits.json`,
`manifest.csv`, `DATA_PROVENANCE.md`, and `code/` (the repo's `src/` and `configs/`). Labels are weak (GLOID polygons, not hand-digitised);
expect IoU on unseen lakes below the 0.90 reference target, and read the per-lake numbers, not just the mean.

## 1. Upload
1. Kaggle > Datasets > New Dataset > upload `glacierguards_kaggle.zip` (Kaggle unzips it). Edit `dataset-metadata.json` is not needed for the web UI.
   Name it `glacierguards-gloid`; keep it private (licence CC BY 4.0 requires attribution, already in `DATA_PROVENANCE.md`).
2. New Notebook > "+ Add Input" > your dataset. Open `kaggle_train.ipynb` from this repo (kaggle/) (File > Import Notebook).

## 2. Notebook settings
- Accelerator: **GPU T4 x1** (or P100). ~16 GB RAM is plenty (tiles are loaded into RAM, about 3 GB).
- Internet: **ON for the first run.** It is needed to `pip install segmentation-models-pytorch` and to download ImageNet encoder weights.
  If you must keep internet off, attach the smp wheels as a dataset and add `--no-pretrained` (expect lower IoU).
- Persistence: "Save & Run All (Commit)" so outputs in `/kaggle/working` survive.

## 3. Commands (run from `/kaggle/working/code`; `DATA` is the dataset dir)
Recommended hyper-parameters: epochs 40, batch size 16, lr 3e-4, AMP on (automatic on GPU), cosine LR, AdamW, early-stopping patience 15 from `pipeline.yaml`.
Loss is BCE + Dice + Focal, summed (weights/gamma/alpha in `configs/pipeline.yaml: model.segmentation.loss`).

| Purpose | Command |
|---|---|
| Sanity run (1 fold) | `python src/training/train_segmentation.py --data-root $DATA --split cv --fold 0 --epochs 40 --batch-size 16 --lr 3e-4` |
| 5-fold lake-grouped CV | repeat with `--fold 1..4` |
| Leave-one-lake-out (one lake) | `... --split lake --holdout-lake GLO_86.47904_27.85858 ...` |
| Final model + held-out test lakes | `... --split test --epochs 40 ...` (trains on the whole pool, evaluates on `test_lakes`) |
| Ablations | add `--channels sar` or `--channels optical` (default `fused`) to any command above |

Run order: fold 0 (check it trains) -> 5-fold CV (fused) -> same CV for `sar` and `optical` -> `--split test` for the fused model.
Full leave-one-lake-out over all pool lakes is ~24 runs; do it for a handful of lakes (the cohort lakes in `splits.json: lolo` and a few of each size class) unless you have quota.

## 4. What is logged and saved
- Per epoch: BCE, Dice, Focal (train) and IoU, F1, precision, recall (validation, pooled over validation lakes).
- At the end: **per held-out lake** IoU / F1 / precision / recall / accuracy in `models/<name>_metrics.json` under `per_lake`, plus the full epoch history.
- Weights (best validation IoU): `models/<name>.pt`, e.g. `seg_fused_cv_fold0.pt`, `seg_fused_test.pt`. Download from the notebook's Output tab,
  or run the last notebook cell which zips `models/` into `results.zip`.

## 5. Interpreting results
- Compare fused vs SAR-only vs optical-only on the same folds; the fused model should win on pooled IoU/F1.
- Weak-label ceiling: GLOID itself reports F1 about 0.82 (S1) and 0.92 (S2), so scores far above that mostly mean "agrees with the label model".
- Small lakes (< 0.2 km2, a few pixels wide at 10 m) will score worst; report by size class using `manifest.csv`.
- South Lhonak is never in any split. After training, run `src/training/predict_masks.py` on it as an inference-only check.
