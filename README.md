# WFM Meningioma T1c Ablation Reconstruction

This repository contains the reconstructed training, sampling, and registered
evaluation code for WFM-based meningioma T1c synthesis experiments.

It accompanies the manuscript:

> **Wavelet Flow Matching for Synthesizing Contrast-Enhanced T1-Weighted MRI from Non-Contrast MRI in Patients with Meningioma: An Image-Level Ablation Study with Internal and External Validation**

The implementation is derived from the original WFM codebase and includes the
three manuscript modifications:

- wavelet-domain high-frequency loss
- DAGCN difference-map guidance
- multiscale anatomical consistency loss

## Important Note on AdaGN

The official WFM training script and the experiments reconstructed here use
additive conditioning injection:

```python
use_scale_shift_norm=False
```

An AdaGN/FiLM-style scale-shift variant is available through
`--use_scale_shift_norm` for exploratory comparison, but it is not the default
setting used by the manuscript ablation models.

## Repository Layout

```text
guided_diffusion/
  wfm.py                         final WFM, DAGCN, and ablation logic
  ablation_losses.py             high-frequency and SSIM helpers
  anatomy_loss.py                manuscript anatomy consistency loss
  meningioma_loader.py           meningioma NIfTI loader
  unet.py                        conditional 3D U-Net
  script_util.py                 model factory
  wfm_original.py               original public WFM reference
  wfm_release_legacy.py         historical DAGCN-era implementation
  wfm_release_pre_dagcn.py      historical pre-DAGCN implementation
  anatomy_loss_legacy.py        historical anatomy-loss implementation
scripts/
  train_wfm_ablation.py         main ablation training entry point
  sample_wfm_ablation.py        sampling entry point
  evaluate_registered.py        manuscript registered evaluation
  statistics_wfm_article.py     paired statistics and BH correction
  run_ablation.sh               12-model main ablation grid
  check_model_variants.py       model parameter/shape checks
docs/
  REPRODUCTION_GUIDE.md         detailed command reference
  CODE_LAYOUT.md                module map
  REPRODUCIBILITY_NOTES.md      implementation caveats and provenance
  WFM_ORIGINAL_README.md        original WFM repository README
DWT_IDWT/                       Haar DWT/IDWT layers
requirements.txt
environment.yml
LICENSE
NOTICE
```

## Data Layout

The loader expects one directory per case:

```text
data/train/
  case_001/
    case_001-t1n.nii.gz
    case_001-t1c.nii.gz
    case_001-t2w.nii.gz
    case_001-t2f.nii.gz
```

`_t1n` and `-t1n` naming are accepted. Each modality is normalized using the
0.1%-99.9% percentile range and resampled to `32 x 512 x 512`.

Patient data are not included in this repository. Users must obtain the data
under the applicable ethics, privacy, and institutional data-use approvals.

## Environment

```bash
conda env create -f environment.yml
conda activate cwdm
```

Or:

```bash
pip install -r requirements.txt
```

## Quick Start

Train one configuration:

```bash
python scripts/train_wfm_ablation.py   --variant anatomy   --data_dir /path/to/data/train   --output_dir runs/anatomy_500   --max_iterations 500
```

Run the full 12-model grid:

```bash
DATA_DIR=/path/to/data/train OUT_ROOT=/path/to/runs bash scripts/run_ablation.sh
```

Include the nine independent supplementary ablations:

```bash
RUN_INDEPENDENT=1 DATA_DIR=/path/to/data/train OUT_ROOT=/path/to/runs bash scripts/run_ablation.sh
```

See [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) for all variants,
sampling, evaluation, and statistics commands.

## Ablation Variants

| Variant | High-frequency | DAGCN | Anatomy |
|---|---:|---:|---:|
| Original | no | no | no |
| HighFreq Pure | yes | no | no |
| DAGCN | yes | yes | no |
| Anatomy | yes | yes | yes |

Independent variants:

- `fm_diff`
- `fm_anat`
- `fm_hf_anat`

## Evaluation

The manuscript evaluation uses `scripts/evaluate_registered.py`:

```bash
python scripts/evaluate_registered.py   --gt_dir /path/to/internal_val   --pred_dir predictions/anatomy_1000   --output_dir metrics/anatomy_1000   --model_tag anatomy_1000   --num_slices 10   --pca_components 64   --fid_feature_mode pca64   --fid_point_estimate median   --bootstrap 1000
```

This performs registration, LPIPS/ROI-MAE/MSE/Dice calculation, ResNet50 feature
extraction, optional PCA-64, and case-level bootstrap FID/MMD estimation.
Paired model comparisons are computed by `scripts/statistics_wfm_article.py`.

## Legacy Code

Historical implementations are retained for provenance and should not be used
as the default training path. The final reconstruction is:

```text
guided_diffusion/wfm.py
guided_diffusion/anatomy_loss.py
guided_diffusion/ablation_losses.py
scripts/train_wfm_ablation.py
scripts/evaluate_registered.py
```

## Original WFM

The original WFM README and citation are retained in
[docs/WFM_ORIGINAL_README.md](docs/WFM_ORIGINAL_README.md).
The original WFM source is by Yalcin Tur, Mihajlo Stojkovic, and Ulas Bagci,
and is distributed under the MIT License.

## License and Attribution

This reconstruction is distributed under the MIT License. See `LICENSE` and
`NOTICE` for the original WFM, cWDM, and third-party attribution.

