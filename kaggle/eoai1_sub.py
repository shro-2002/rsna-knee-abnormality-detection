"""E-OAI-1 exploratory submission (research/37; owner-approved 2026-10-16, one submission, NOT a final selection).
Blend frozen before any run: 0.5 x rank(E-DIV equal-weight blend of 4 arms) + 0.5 x rank(nartaa OAI arm), per target.
ATTRIBUTION (third-party weights, used under their terms; no copies redistributed):
  dreaddevelopment: Raptor CoAtNet v5/v10/v9 (CC0-1.0); goodpjw2008: 2.5D ConvNeXt reader + knee.py/preprocess.py (Apache-2.0).
  nartaa (Danial Zakaria): rsna-knee-publication-swa-weights-20261007 v1, raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt,
     sha256 7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef; "research and educational use, including the RSNA
     Knee competition" (dataset terms). Inference geometry follows nartaa's public notebook rsna-knee-0949-anatomical-mirror.
  OAI acknowledgement (required by the weights' terms): Data and/or research tools used in the preparation of this manuscript were
     obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data
     repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National
     Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a
     worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development.
     Dataset identifier(s): 10.15154/0hcg-f676. We did not access OAI data; only the public learned checkpoint is used.
OURS: stack builder, model loader, mirror/crop TTA, sharded runner and blend (private dataset sonika20000002/rsna-knee-2026-eoai-src).
"""
import glob, os, shutil, subprocess, sys

def find(marker):
    for d in range(1, 7):
        h = glob.glob("/kaggle/input/" + "/".join(["*"] * d) + "/" + marker)
        if h: return h[0]
# the private bundle may mount with or without its top-level "src/" folder: copy the package to a fixed place as "src"
pkg = find("ediv/run_ensemble.py")[:-len("/ediv/run_ensemble.py")]
bundle = "/tmp/eoai_bundle"
shutil.copytree(pkg, bundle + "/src", dirs_exist_ok=True)
print("EOAI1 package", pkg, "->", bundle, flush=True)
os.chdir(bundle); sys.path.insert(0, bundle)
import pandas as pd
from src.ediv.run_ensemble import comp_root
ARMS = "raptor_v5,raptor_v10,raptor_v9,convnext,nartaa_oai"
run = lambda mode, deadline: subprocess.run([sys.executable, "-m", "src.ediv.run_ensemble", "--mode", mode, "--blend", "eoai",
                                             "--arms", ARMS, "--out", "/kaggle/working", "--gpus", "2", "--workers", "2",
                                             "--deadline", str(deadline)]).returncode
# Commit run only (visible test = 3 studies): GPU preflight on gold-58 + 120-study bench, checked offline BEFORE submitting.
# The hidden-test rerun skips this and only scores the test set.
if len(pd.read_csv(f"{comp_root()}/test.csv")) <= 10:
    for mode in ("gold", "bench"):
        assert run(mode, 2400) == 0, f"preflight {mode} failed"
assert run("test", 7.0 * 3600) == 0 and os.path.exists("/kaggle/working/submission.csv"), "ensemble run failed"
print("EOAI1_SUBMISSION_OK")
