# Experiment record

All runs: Kaggle free GPUs (T4 / 2 × T4), $0 spend. "Gold-58" = the 58 training studies with radiologist labels for all 12
findings. It is tiny (bootstrap 95 % CI about ±0.05) and several public checkpoint authors used it for model selection, so it is
a diagnostic, not a clean validation score. "Weak-CV" = study-level 5-fold cross-validation on 4,349 studies against weak
report labels (CI about ±0.02); it measures fit to noisy labels and was used for selection only. Public LB = Kaggle public
leaderboard, part of the test set.

## Scored submissions

| # | Date | Kaggle ref | Model | Weights | Gold-58 | Public LB |
|---|---|---|---|---|---|---|
| 1 | 10-12 | 57118112 | EXP-004: frozen EfficientNet-B0, 2.5D, gated-attention MIL, 224 px / 8 slices | own heads (ImageNet backbone) | 0.620 (0.578–0.661) | 0.587 |
| 2 | 10-13 | 57146756 | ENS-01: rank blend EXP-004 + EXP-005 (frozen DINOv2 ViT-S/14) | own | 0.638 (0.588–0.685) | 0.605 |
| 3 | 10-14 | 57182192 | ENS-02: same arms, heads retrained on a consensus of public LLM-read report labels | own | 0.737 (1 seed) | **0.750** |
| 4 | 10-14 | 57184755 | Raptor v4: another author's public notebook run unchanged (benchmark) | third-party | 0.913 | 0.924 |
| 5 | 10-14 | 57190635 | **E-DIV** (final): own inference code, equal rank blend of 4 public checkpoints | third-party | 0.927 | **0.938** |
| 6 | 10-17 | 57300241 | **E-OAI-1** (final): 0.5 × rank(E-DIV) + 0.5 × rank(nartaa OAI-trained CoAtNet) | third-party | 0.931 (diagnostic) | **0.947** |

E-OAI-1 had a pre-declared promotion gate of 0.953 public and missed it (0.947). It is a final because it is the best measured
score; I selected both finals.

## Own models not submitted

| ID | Hypothesis | Result | Decision |
|---|---|---|---|
| EXP-001/002/008 | Lexicon + negation extractor gives usable weak labels | soft-label macro AUC vs gold 0.652 → 0.667 → 0.689 (Spanish, Dutch, English, Greek; OA phrasing) | kept as a label source |
| EXP-005 | Frozen DINOv2 resolves focal lesions EfficientNet cannot | 0.611 vs 0.620, CIs overlap; complementary per target | kept as ensemble arm only |
| EXP-007 | True 2.5D (adjacent slices as RGB) breaks the frozen plateau | 0.599, twice the runtime | rejected |
| EXP-006 | LoRA fine-tune (last 4 blocks, r = 8) breaks the plateau | 0.603; loss fell 3 % in 5 epochs (underfit) | rejected |
| HEAD-01 | 12 per-finding attention heads beat one shared head | worse on weak-CV and gold for both backbones | rejected |
| E1 | DINOv2 ViT-S full fine-tune, 3 planes × 16 slices, 224 px, 2 × T4 DDP | gold-58 0.879 (0.845–0.908); missed pre-declared gate | stopped |
| E1-HR | 100 mm joint-centred crop + dense 2.5D dual view | gold-58 0.890; paired vs E1 +0.011 (CI low −0.006); missed gates | stopped |
| FUSE-NX | Cross-fitted fused labels (0.5 report + 0.5 out-of-fold image teacher) | labels: 0.921 vs 0.876 vs gold; student model: Δ −0.0003 (90 % CI −0.0072 to +0.0069) | route closed |

## Lessons

1. **Measure the representation before tuning the head.** The first model scored below its own training labels. That pointed
   at the frozen 224 px backbone, and four cheap head-side changes confirmed a plateau at 0.60–0.62.
2. **Label quality moved my own models most.** ENS-02 changed only the training labels and gained +0.145 public LB.
3. **A model cannot beat the agreement of its labels with the truth by much.** Full fine-tunes stopped near 0.88–0.89 on gold,
   just above the labels' own 0.876.
4. **Label gains do not transfer automatically.** FUSE-NX produced clearly better labels and no better model; the student
   already extracted the image signal the teacher added. A label-level gain is not a proxy for a model gain.
5. **Pre-declare gates and keep the negative results.** Every GPU run had a written gate before launch. E1, E1-HR, FUSE-NX and
   E-OAI-1 all missed theirs, and are reported here.
6. **Do not calibrate on tiny sets.** "LB = gold-58 − 0.033" held for two submissions and broke on the third (+0.013).
7. **Engineering details carried the third-party models.** Correct series selection, geometry-based slice ordering, shared
   decoding and per-study failure flags were what made four public checkpoints run as one offline 9-hour submission.

## Gaps a reviewer should know about

- Patient-level leakage across weak-CV folds is UNKNOWN: the data has no patient ID.
- Third-party checkpoint training data is as reported by their authors (no logs). The nartaa checkpoint used OAI data.
- Training code for my own models is here; the Kaggle kernel builders and run logs are not.
- Private leaderboard results are not available until after 2026-10-22.
