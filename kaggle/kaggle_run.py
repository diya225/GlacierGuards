# ===== Kaggle notebook: paste each "# %%" block into its own cell =====
# Settings: Accelerator = GPU T4/P100, Internet = ON (for pretrained weights + pip).
# Add your uploaded dataset (glof-ews) via "+ Add Input".

# %% 1. setup
import subprocess, glob, shutil, os
zip_path = glob.glob("/kaggle/input/**/glof_ews_kaggle.zip", recursive=True)
if zip_path:                                     # dataset contains the zip
    shutil.unpack_archive(zip_path[0], "/kaggle/working/glof")
else:                                            # Kaggle auto-extracted the dataset
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/src/__init__.py", recursive=True)[0]))
    shutil.copytree(root, "/kaggle/working/glof", dirs_exist_ok=True)
os.chdir("/kaggle/working/glof")
subprocess.run("pip -q install segmentation-models-pytorch rasterio pymannkendall pyproj", shell=True)

# %% 2. train fused model (date-held-out split). Add --lakes a b c once more lakes are labeled.
!python src/training/train_segmentation.py --epochs 60 --split date --lr 5e-4 --batch-size 16
# optional ablations (SAR-only / optical-only):
# !python src/training/train_segmentation.py --epochs 60 --channels sar --lr 5e-4 --batch-size 16
# !python src/training/train_segmentation.py --epochs 60 --channels optical --lr 5e-4 --batch-size 16

# %% 3. predict masks for ALL dates, then temporal model + residual plot
!python src/training/predict_masks.py --lake south_lhonak
!python src/training/train_temporal.py --loo
!python src/training/plot_anomaly.py

# %% 4. collect outputs: download /kaggle/working/glof/models, data/predictions, reports
!zip -qr /kaggle/working/results.zip models data/predictions reports
