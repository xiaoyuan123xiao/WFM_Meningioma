# WFM Meningioma T1c Ablation Reconstruction

This project reconstructs the WFM ablation experiments from:

- the original WFM paper and source repository
- the final meningioma ablation manuscript
- `WFM_release_20260920.zip`, which contains the recovered experimental modules
- the user-provided summary of the three modifications

The implementation defaults to the **final manuscript formulas**. Compatibility
switches are provided for the recovered release-code variant.

## Implemented Modules

### WFM backbone

`guided_diffusion/wfm.py` implements 3D Haar DWT/IDWT, the wavelet-space informed
source, linear flow matching, velocity prediction, class conditioning, and
direct/Euler/Heun inference.  Input channels are 32 for the baseline and 33 when
DAGCN is enabled.

### High-frequency loss

The training loop uses the manuscript formula:

```python
pred_target_dwt = source_dwt + model_output
hf_loss = mse(pred_target_dwt[:, 1:], target_dwt[:, 1:])
```

The loss constrains predicted target high-frequency subbands, not the raw
velocity channels.

### DAGCN discrepancy guidance

The discrepancy predictor is:

```text
Conv3d(24, 64, 3, padding=1) -> BatchNorm3d -> ReLU
Conv3d(64, 32, 3, padding=1) -> BatchNorm3d -> ReLU
Conv3d(32,  1, 3, padding=1) -> Sigmoid
```

Default manuscript behavior:

- target map: mean absolute difference of the seven high-frequency subbands
- model input: `concat(x_t, cond_dwt, predictor(cond_dwt))`
- predictor loss: MSE

Recovered release-code behavior can be selected with:

```text
--discrepancy_target lll_abs --discrepancy_input target
```

This uses the LLL absolute difference and the detached ground-truth map as the
training-time 33rd channel.

### Anatomy consistency loss

`guided_diffusion/anatomy_loss.py` defaults to the manuscript architecture:

```text
Conv3d(3x3x3) -> InstanceNorm3d -> ReLU -> MaxPool3d(2x2x2)
channels: 16 / 32 / 64 / 128
random initialization, frozen parameters
```

The recovered release implementation is available through:

```text
--anatomy_arch legacy
```

## Data Layout

The loader expects one directory per case:

```text
train_root/
  case_001/
    case_001-t1n.nii.gz
    case_001-t1c.nii.gz
    case_001-t2w.nii.gz
    case_001-t2f.nii.gz
```

`_t1n` naming is also accepted.  All modalities are normalized independently
with the 0.1%-99.9% percentile range and resampled to `32 x 512 x 512`.

The same random flip and in-plane rotation is applied to all four modalities.

## Environment

Use the original WFM environment:

```bash
conda env create -f environment.yml
conda activate wfm
pip install -r requirements.txt
```

The current reconstruction uses PyTorch, nibabel, pywavelets, tensorboard, and
tqdm.

## Train One Configuration

Baseline:

```bash
python scripts/train_wfm_ablation.py \
  --variant original \
  --data_dir /data/meningioma/train \
  --output_dir runs/original_500 \
  --max_iterations 500
```

High-frequency loss:

```bash
python scripts/train_wfm_ablation.py \
  --variant highfreq_pure \
  --data_dir /data/meningioma/train \
  --output_dir runs/highfreq_500 \
  --max_iterations 500
```

DAGCN, manuscript version:

```bash
python scripts/train_wfm_ablation.py \
  --variant dagcn \
  --data_dir /data/meningioma/train \
  --output_dir runs/dagcn_500 \
  --max_iterations 500
```

DAGCN, recovered release version:

```bash
python scripts/train_wfm_ablation.py \
  --variant dagcn \
  --data_dir /data/meningioma/train \
  --output_dir runs/dagcn_legacy_500 \
  --max_iterations 500 \
  --discrepancy_target lll_abs \
  --discrepancy_input target
```

Anatomy consistency:

```bash
python scripts/train_wfm_ablation.py \
  --variant anatomy \
  --data_dir /data/meningioma/train \
  --output_dir runs/anatomy_500 \
  --max_iterations 500
```

## Train the 12-Model Ablation Grid

```bash
DATA_DIR=/data/meningioma/train \
OUT_ROOT=/data/runs/wfm_meningioma \
bash scripts/run_ablation.sh
```

This runs four configurations at 200/500/1000 training iterations, matching
the 12 seed-42 models in the manuscript. The script defaults to SSIM_WEIGHT=0.0, following the manuscript where Original uses only flow matching. Set it to 0.1 only to reproduce the historical summary variant.

## Sampling

The manuscript evaluates training budgets, not sampling budgets.  The best WFM
inference setting is direct single-step prediction:

```bash
python scripts/sample_wfm_ablation.py \
  --variant anatomy \
  --checkpoint runs/anatomy_1000/checkpoints/step_001000.pt \
  --data_dir /data/meningioma/internal_val \
  --output_dir predictions/anatomy_1000
```

For a two-step Euler or Heun run:

```bash
python scripts/sample_wfm_ablation.py \
  --variant anatomy \
  --checkpoint runs/anatomy_1000/checkpoints/step_001000.pt \
  --data_dir /data/meningioma/internal_val \
  --output_dir predictions/anatomy_1000_heun \
  --sampling_steps 2 \
  --solver heun
```

## Discrepancies Recovered From the Old Code

The final manuscript and the recovered release code differ in these details:

| Item | Manuscript | Recovered release code |
|---|---|---|
| DAGCN target | mean absolute difference of 7 HF bands | absolute LLL difference |
| DAGCN 33rd input at train time | predictor output | detached ground-truth map |
| anatomy extractor | Conv+InstanceNorm+ReLU+MaxPool | Conv(stride=2)+ReLU |
| high-frequency target | `source + velocity` | some experiment scripts used velocity directly |

The CLI switches preserve both versions:

```text
--high_freq_prediction {pred_target,velocity}
--discrepancy_target {hf_mean,lll_abs,image_diff}
--discrepancy_input {predicted,target}
--anatomy_arch {paper,legacy}
```

## Checkpoints

Each checkpoint contains:

- `model`: 3D U-Net state dict
- `wfm`: discrepancy-predictor state dict, if present
- `optimizer` and `scheduler`
- `args`: exact training configuration
- `step`: training iteration

The sample script can also load a model-only state dict for compatibility with
the original WFM repository.



## Reference Implementations Kept in the Repository

- `guided_diffusion/wfm_original.py`: original public WFM code
- `guided_diffusion/wfm_release_legacy.py`: DAGCN-era release code
- `guided_diffusion/wfm_release_pre_dagcn.py`: pre-DAGCN experimental backup
- `guided_diffusion/anatomy_loss_legacy.py`: recovered legacy anatomy extractor


## Article-Aligned Evaluation

`scripts/evaluate.py` is the original WFM evaluator and is not the manuscript
evaluation protocol.  Use `scripts/evaluate_registered.py` for the manuscript:

```bash
python scripts/evaluate_registered.py \
  --gt_dir /data/meningioma/internal_val \
  --pred_dir predictions/anatomy_1000 \
  --output_dir metrics/anatomy_1000 \
  --model_tag anatomy_1000 \
  --num_slices 10 \
  --pca_components 64 \
  --fid_feature_mode pca64 \
  --fid_point_estimate median \
  --bootstrap 1000
```

This script:

- resamples GT and prediction to `32 x 512 x 512`
- performs ANTs rigid registration
- computes per-case LPIPS, ROI-MAE, MSE, and brain-mask Dice
- extracts ResNet50 slice features
- uses raw 2048-d slice features by default, or fits PCA-64 when requested
- performs case-level bootstrap for FID/MMD median and 95% CI
- writes `per_case_registered_<tag>.csv` and
  `metrics_registered_<tag>.csv`

For paired model comparisons:

```bash
python scripts/statistics_wfm_article.py \
  --results_dir metrics \
  --baseline original_500 \
  --output_csv pairwise_wilcoxon.csv
```

This computes paired Wilcoxon tests, bootstrap 95% CIs for paired mean
differences, Cliff's delta, and Benjamini-Hochberg q-values.

## Manuscript Ablation Loss Combinations

The main manuscript uses a cumulative design:

| Variant | High-frequency | DAGCN | Anatomy |
|---|---:|---:|---:|
| Original | no | no | no |
| HighFreq Pure | yes | no | no |
| DAGCN | yes | yes | no |
| Anatomy | yes | yes | yes |

The supplementary independent variants are also available:

- `fm_diff`: FM + discrepancy loss
- `fm_anat`: FM + anatomy loss
- `fm_hf_anat`: FM + high-frequency + anatomy

The default high-frequency implementation follows the manuscript equation at
`t=0`: `x_source + v(x_source, c, 0, y)`.  Use `--high_freq_time current` only
for the older experimental implementation.

The default model uses additive conditioning injection
(`use_scale_shift_norm=False`), matching the official WFM code and the
experiments. Use `--use_scale_shift_norm` only for exploratory AdaGN variants.

For FID/MMD, the manuscript contains two reported protocols:

- `--fid_feature_mode raw2048`: 10 slice-level ResNet50 features per case,
  used for the main ablation tables.
- `--fid_feature_mode pca64`: fit PCA to 64 components before FID/MMD, used for
  the reviewer-response bootstrap tables (S7/S8).

Both modes perform case-level bootstrap with `B=1000` and report the median and
95% CI, matching the result summaries.

Set `RUN_INDEPENDENT=1` when invoking `run_ablation.sh` to additionally train
the nine supplementary models: `diff`, `anat`, and `hf_anat` at
200/500/1000 iterations.



## Archived Sampling Output Format

The archived `dagcn_200` output uses:

- one directory per case, e.g. `YL301/`
- `sample_t1c.nii.gz`
- `target_t1c.nii.gz`
- shape `32 x 512 x 512`
- `float32`
- unit pixdim and identity affine

`sample_wfm_ablation.py` now matches this format and saves the target by default.

