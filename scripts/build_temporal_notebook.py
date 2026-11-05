"""Writes kaggle/kaggle_temporal.ipynb (Milestone 6 on Kaggle). Run once: python scripts/build_temporal_notebook.py"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip().splitlines(True)}


cells = [
    md("""# GlacierGuards Milestone 6: temporal anomaly check on South Lhonak
Needs the Kaggle dataset `glacierguards-temporal` (built by `scripts/package_temporal_for_kaggle.py`, about 70 MB) attached as an input.
Accelerator: GPU T4 (the script uses CUDA automatically). Internet is NOT needed (only torch, numpy, pandas, pyyaml, matplotlib)."""),
    code("""
import glob, os, shutil, subprocess, sys
src = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/src/training/train_temporal.py', recursive=True)[0]))
src = os.path.dirname(src)  # dataset root that contains src/, configs/, data/
print('dataset root:', src)
shutil.copytree(src, '/kaggle/working/gg', dirs_exist_ok=True)
os.chdir('/kaggle/working/gg')
import torch
print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print(sorted(os.listdir('data/predictions/south_lhonak'))[:3], '...', len(os.listdir('data/predictions/south_lhonak')), 'files')
"""),
    md("""## 1. Simple baseline (no training): robust drop detector
This is the bar the forecaster has to beat. It must show the burst on 2023-10-06 and no false alarms before it."""),
    code("!python src/training/burst_detection.py --lake south_lhonak --burst-date 2023-10-04"),
    md("""## 2. ConvLSTM forecaster, scored out-of-sample
Blocked CV by season (each season is scored by a model that never saw it) and trained only on dates before the burst.
Watch the log: it prints the time of the first fit, then one line per held-out season."""),
    code("""
!python -u src/training/train_temporal.py --lakes south_lhonak --cv-year --train-before 2023-10-04 --kind conv_lstm --hidden 16 --epochs 40 --tag _convlstm 2>&1 | tail -45
"""),
    md("## 3. Attention forecaster (second model, same protocol)"),
    code("""
!python -u src/training/train_temporal.py --lakes south_lhonak --cv-year --train-before 2023-10-04 --kind attention --hidden 16 --epochs 40 --tag _attn 2>&1 | tail -45
"""),
    md("""## 4. Compare with persistence (the same-as-last-scene baseline)
A forecaster only adds value if its pre-burst out-of-sample error is lower than persistence and the burst still stands out."""),
    code("""
import pandas as pd, numpy as np
for tag in ['_convlstm', '_attn']:
    d = pd.read_csv(f'data/predictions/south_lhonak/anomaly_series{tag}.csv', parse_dates=['date'])
    pre = d[d.date < '2023-10-04']; post = d[d.date >= '2023-10-04']
    mae_f = (pre.observed_area_m2 - pre.predicted_area_m2).abs().mean() / 1e6
    mae_p = (pre.observed_area_m2 - pre.persistence_area_m2).abs().mean() / 1e6
    print(f'{tag}: pre-burst MAE forecaster {mae_f:.3f} km2 vs persistence {mae_p:.3f} km2 (n={len(pre)})')
    print('  most negative residual z after the burst:', post.residual_z.min().round(2), 'on', post.loc[post.residual_z.idxmin(), 'date'].date(),
          '| pre-burst |z|>4:', int((pre.residual_z.abs() > 4).sum()), 'of', len(pre))
"""),
    code("""
os.makedirs('reports', exist_ok=True)
shutil.copy('data/predictions/south_lhonak/anomaly_series_convlstm.csv', 'data/predictions/south_lhonak/anomaly_series.csv')  # plot_anomaly reads this name
print(subprocess.run([sys.executable, 'src/training/plot_anomaly.py', '--lake', 'south_lhonak'], capture_output=True, text=True).stdout[-300:])
"""),
    md("## 5. Save outputs (download from the Output tab)"),
    code("""
import zipfile, glob
with zipfile.ZipFile('/kaggle/working/temporal_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for pat in ['data/predictions/south_lhonak/anomaly_series*.csv', 'models/temporal_checkpoint*.pt', 'reports/*']:
        for f in glob.glob(pat):
            z.write(f)
print(zipfile.ZipFile('/kaggle/working/temporal_results.zip').namelist())
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = R / "kaggle" / "kaggle_temporal.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
