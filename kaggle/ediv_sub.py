"""E-DIV submission: equal-weight per-target rank blend of four public, OAI-free readers.
ATTRIBUTION (all weights/code below are third-party and used under their licences; see docs/EDIV_PROVENANCE.md in our repo):
  dreaddevelopment (Kaggle): Raptor CoAtNet checkpoints v5/v10/v9 (CC0-1.0): raptor-knee-maxspan, -native384dense, -finespacing;
     geometry/preprocessing specification from their public notebook 'Knee MRI: twelve findings from a single model' (Apache-2.0).
  goodpjw2008 (Kaggle): 2.5D ConvNeXt-MIL reader weights + knee.py/preprocess.py (Apache-2.0): rsna-knee-2-5d-convnext-reader.
  timm / Hugging Face: backbone architectures.
OURS: independent stack builder, model loader, sharded 2-GPU runner, validation and blend (src/ediv, src/ensemble) in dataset
  sonika20000002/rsna-knee-2026-src. No weights were trained or tuned by us; no OAI-derived weights; blend weights fixed in advance.
"""
import glob, os, subprocess, sys

def find(marker):
    for d in range(1, 7):
        h = glob.glob("/kaggle/input/" + "/".join(["*"] * d) + "/" + marker)
        if h: return h[0]
bundle = find("src/ediv/run_ensemble.py")[:-len("/src/ediv/run_ensemble.py")]
os.chdir(bundle)
r = subprocess.run([sys.executable, "-m", "src.ediv.run_ensemble", "--mode", "test", "--out", "/kaggle/working", "--gpus", "2", "--workers", "2"])
assert r.returncode == 0 and os.path.exists("/kaggle/working/submission.csv"), "ensemble run failed"
print("EDIV_SUBMISSION_OK")
