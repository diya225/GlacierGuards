"""Phase 4 check: unzip glacierguards_kaggle.zip into a clean dir and run 2 training steps (--split lake) in a fresh process."""
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
d = Path(tempfile.mkdtemp(prefix="gg_verify_", dir=R / "data"))
zipfile.ZipFile(R / "glacierguards_kaggle.zip").extractall(d)
sp = json.loads((d / "splits.json").read_text())
assert not any("lhonak" in l.lower() for l in sp["train_pool"] + sp["test_lakes"])
assert not set(sp["train_pool"]) & set(sp["test_lakes"]), "lake overlap train/test"
tiles = list((d / "tiles").glob("*.npz"))
assert tiles, "no tiles"
for f in tiles:
    z = np.load(f)
    assert np.isfinite(z["x"]).all() and np.isfinite(z["elev"]).all(), f"NaN in {f.name}"
    assert z["y"].shape[0] == z["x"].shape[0] == z["valid"].shape[0] == z["elev"].shape[0], f"misaligned {f.name}"
cmd = [sys.executable, "-I", str(d / "code" / "src" / "training" / "train_segmentation.py"), "--data-root", str(d), "--split", "lake",
       "--holdout-lake", sp["lolo"][0], "--epochs", "1", "--max-steps", "2", "--batch-size", "4", "--no-pretrained"]
r = subprocess.run(cmd, cwd=d / "code", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else "")
print("verified OK" if r.returncode == 0 else "VERIFY FAILED", "in", d)
sys.exit(r.returncode)
