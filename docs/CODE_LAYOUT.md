# Code Layout

## Main training and model path

- `guided_diffusion/wfm.py`
  - Haar DWT/IDWT flow-matching implementation
  - informed source prior
  - class/time conditioning
  - DAGCN discrepancy predictor
  - final ablation inference paths
- `guided_diffusion/unet.py`
  - conditional 3D U-Net
  - additive conditioning by default (`use_scale_shift_norm=False`)
  - optional FiLM/AdaGN-style scale-shift exploration
- `guided_diffusion/script_util.py`
  - model factory and model argument definitions
- `guided_diffusion/meningioma_loader.py`
  - meningioma NIfTI dataset loader
  - normalization, resampling, augmentation
- `guided_diffusion/ablation_losses.py`
  - SSIM and high-frequency helpers
- `guided_diffusion/anatomy_loss.py`
  - multi-scale anatomical consistency loss

## Entry points

- `scripts/train_wfm_ablation.py`
  - main training script for the 12-model ablation grid and independent ablations
- `scripts/sample_wfm_ablation.py`
  - inference and sampling for the reconstructed model variants
- `scripts/evaluate_registered.py`
  - manuscript-level registered evaluation
- `scripts/statistics_wfm_article.py`
  - paired Wilcoxon tests, Cliff's delta, bootstrap confidence intervals, BH correction
- `scripts/run_ablation.sh`
  - main cumulative ablation grid
- `scripts/check_model_variants.py`
  - parameter-count and optional forward-shape validation

## Historical code

These files are retained for provenance and are not part of the default final
training path:

- `guided_diffusion/wfm_original.py`
- `guided_diffusion/wfm_release_legacy.py`
- `guided_diffusion/wfm_release_pre_dagcn.py`
- `guided_diffusion/anatomy_loss_legacy.py`
- `scripts/evaluate_registered_user_original.py`

## Data and outputs

Patient data, checkpoints, TensorBoard events, predictions, and metric outputs
are intentionally excluded from version control by `.gitignore`.
