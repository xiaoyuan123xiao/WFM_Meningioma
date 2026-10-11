"""Article-aligned post-registration evaluation for WFM meningioma T1c synthesis.

Pipeline:
1. resample GT and prediction to 32 x 512 x 512
2. rigidly register the prediction to the GT with ANTs
3. normalize both images to [0, 1]
4. compute per-case LPIPS, ROI-MAE, MSE, and brain-mask Dice
5. extract ResNet50 slice features, fit PCA-64, and compute FID/MMD
6. bootstrap cases to obtain median and 95% CI for FID/MMD

This is the corrected version of the user-supplied evaluate_registered.py.
"""

import argparse
import glob
import os
import warnings
from pathlib import Path

import ants
import lpips
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.linalg import sqrtm
from scipy.ndimage import binary_fill_holes, label
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

TARGET_SHAPE = (32, 512, 512)
FEATURE_DIM = 2048

def normalize_to_0_1(image):
    low = np.quantile(image, 0.001)
    high = np.quantile(image, 0.999)
    if high - low < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(image, low, high)
    return ((image - low) / (high - low)).astype(np.float32)


def to_dhw(volume):
    """Convert (H, W, D) or (D, H, W) to (D, H, W)."""
    if volume.shape[0] <= 64 or (volume.shape[0] <= volume.shape[1] and volume.shape[0] <= volume.shape[2]):
        return volume
    return np.transpose(volume, (2, 0, 1))


def interpolate_to_target(volume_dhw, target_shape=TARGET_SHAPE):
    if volume_dhw.shape == target_shape:
        return volume_dhw.astype(np.float32)
    tensor = torch.from_numpy(volume_dhw).float().unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return tensor[0, 0].numpy().astype(np.float32)


def ants_rigid_register(fixed_dhw, moving_dhw):
    """Register moving_dhw to fixed_dhw with a 6-DOF ANTs rigid transform."""
    fixed_hwd = np.transpose(fixed_dhw, (1, 2, 0))
    moving_hwd = np.transpose(moving_dhw, (1, 2, 0))
    fixed = ants.from_numpy(fixed_hwd)
    moving = ants.from_numpy(moving_hwd)
    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )
    warped_hwd = registration["warpedmovout"].numpy()
    return np.transpose(warped_hwd, (2, 0, 1)).astype(np.float32)


def get_brain_mask(image, threshold=0.1):
    mask = image > threshold
    mask = binary_fill_holes(mask)
    labeled, num = label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labeled == (int(np.argmax(sizes[1:])) + 1)


class MedicalResNetFeatureExtractor(nn.Module):
    """ResNet50 penultimate features used for FID/MMD."""

    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return features.flatten(1)

def compute_case_metrics(gt_norm, pred_norm, lpips_model, extractor, device, num_slices=10):
    """Compute per-case image metrics and one ResNet50 feature block."""
    gt_mask = get_brain_mask(gt_norm, threshold=0.1)
    pred_mask = get_brain_mask(pred_norm, threshold=0.1)

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = gt_mask.sum() + pred_mask.sum()
    dice = (2.0 * intersection / union) if union > 0 else 0.0

    roi_mae = float(np.mean(np.abs(gt_norm[gt_mask] - pred_norm[gt_mask]))) if gt_mask.sum() > 0 else np.nan
    mse = float(np.mean((gt_norm - pred_norm) ** 2))

    mid = gt_norm.shape[0] // 2
    gt_slice = torch.from_numpy(gt_norm[mid]).float().unsqueeze(0).unsqueeze(0) * 2.0 - 1.0
    pred_slice = torch.from_numpy(pred_norm[mid]).float().unsqueeze(0).unsqueeze(0) * 2.0 - 1.0
    gt_slice = gt_slice.repeat(1, 3, 1, 1).to(device)
    pred_slice = pred_slice.repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        lpips_value = float(lpips_model(gt_slice, pred_slice).item())

    indices = np.linspace(0, gt_norm.shape[0] - 1, int(num_slices), dtype=int)
    gt_features = []
    pred_features = []
    for index in indices:
        t_gt = torch.from_numpy(gt_norm[index]).float().unsqueeze(0).unsqueeze(0) * 2.0 - 1.0
        t_pred = torch.from_numpy(pred_norm[index]).float().unsqueeze(0).unsqueeze(0) * 2.0 - 1.0
        t_gt = t_gt.repeat(1, 3, 1, 1).to(device)
        t_pred = t_pred.repeat(1, 3, 1, 1).to(device)
        gt_features.append(extractor(t_gt).cpu().numpy().reshape(-1))
        pred_features.append(extractor(t_pred).cpu().numpy().reshape(-1))

    return {
        "LPIPS": lpips_value,
        "ROI_MAE": roi_mae,
        "MSE": mse,
        "dice": float(dice),
        "gt_features": np.stack(gt_features, axis=0),
        "pred_features": np.stack(pred_features, axis=0),
    }


def compute_fid(real_features, fake_features):
    mu_real = real_features.mean(axis=0)
    mu_fake = fake_features.mean(axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    if np.ndim(sigma_real) == 0:
        sigma_real = np.array([[float(sigma_real)]])
        sigma_fake = np.array([[float(sigma_fake)]])
    eps = 1e-6
    sigma_real = sigma_real + eps * np.eye(sigma_real.shape[0])
    sigma_fake = sigma_fake + eps * np.eye(sigma_fake.shape[0])
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean))


def _pairwise_sq_dists(x, y):
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True)
    return np.maximum(x_norm + y_norm.T - 2.0 * x @ y.T, 0.0)


def compute_mmd(real_features, fake_features, sigma=None):
    xx = _pairwise_sq_dists(real_features, real_features)
    yy = _pairwise_sq_dists(fake_features, fake_features)
    xy = _pairwise_sq_dists(real_features, fake_features)
    if sigma is None:
        all_dists = np.concatenate([xx.ravel(), yy.ravel(), xy.ravel()])
        sigma = float(np.median(np.sqrt(all_dists)) + 1e-8)
    k_xx = np.exp(-xx / (2.0 * sigma ** 2))
    k_yy = np.exp(-yy / (2.0 * sigma ** 2))
    k_xy = np.exp(-xy / (2.0 * sigma ** 2))
    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))

def bootstrap_mean_ci(values, num_bootstrap=1000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if num_bootstrap <= 0:
        return float(values.mean()), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(num_bootstrap)]
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_feature_distances(real_case_features, fake_case_features, num_bootstrap=1000, seed=42, pca_components=64, feature_mode="pca64", point_estimate="point"):
    """Bootstrap cases and compute raw-2048 or PCA-64 FID/MMD."""
    num_cases = len(real_case_features)
    if num_cases == 0:
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)

    real_full = np.vstack(real_case_features)
    fake_full = np.vstack(fake_case_features)
    if feature_mode == "pca64":
        all_full = np.vstack([real_full, fake_full])
        n_components = min(int(pca_components), all_full.shape[0] - 1, all_full.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(all_full)
        real_case_features = [pca.transform(features) for features in real_case_features]
        fake_case_features = [pca.transform(features) for features in fake_case_features]
    elif feature_mode != "raw2048":
        raise ValueError("feature_mode must be 'raw2048' or 'pca64'")

    real_pca = real_case_features
    fake_pca = fake_case_features
    def point_values(indices):
        real = np.vstack([real_pca[i] for i in indices])
        fake = np.vstack([fake_pca[i] for i in indices])
        sigma = float(np.median(np.sqrt(np.concatenate([
            _pairwise_sq_dists(real, real).ravel(),
            _pairwise_sq_dists(fake, fake).ravel(),
            _pairwise_sq_dists(real, fake).ravel(),
        ]))) + 1e-8)
        return compute_fid(real, fake), compute_mmd(real, fake, sigma=sigma)

    all_indices = np.arange(num_cases)
    fid_point, mmd_point = point_values(all_indices)
    if num_bootstrap <= 0:
        return (fid_point, np.nan, np.nan), (mmd_point, np.nan, np.nan)

    rng = np.random.default_rng(seed)
    fid_values = []
    mmd_values = []
    for _ in range(num_bootstrap):
        indices = rng.integers(0, num_cases, num_cases)
        fid_value, mmd_value = point_values(indices)
        fid_values.append(fid_value)
        mmd_values.append(mmd_value)

    fid_center = fid_point if point_estimate == "point" else float(np.median(fid_values))
    mmd_center = mmd_point if point_estimate == "point" else float(np.median(mmd_values))
    return (
        (float(fid_center), float(np.percentile(fid_values, 2.5)), float(np.percentile(fid_values, 97.5))),
        (float(mmd_center), float(np.percentile(mmd_values, 2.5)), float(np.percentile(mmd_values, 97.5))),
    )

def find_gt_file(case_dir):
    target = os.path.join(case_dir, "target_t1c.nii.gz")
    if os.path.exists(target):
        return target
    files = glob.glob(os.path.join(case_dir, "*t1c*.nii.gz"))
    files = [path for path in files if "sample" not in os.path.basename(path).lower() and "target" not in os.path.basename(path).lower()]
    if not files:
        files = glob.glob(os.path.join(case_dir, "*.nii.gz"))
    return files[0] if files else None


def find_pred_file(case_dir):
    direct = os.path.join(case_dir, "sample_t1c.nii.gz")
    if os.path.exists(direct):
        return direct
    files = glob.glob(os.path.join(case_dir, "*t1c*.nii.gz"))
    files = [path for path in files if "target" not in os.path.basename(path).lower()]
    return files[0] if files else None

def create_argparser():
    parser = argparse.ArgumentParser(description="Article-aligned WFM post-registration evaluation")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth case directories")
    parser.add_argument("--pred_dir", required=True, help="Prediction case directories")
    parser.add_argument("--output_dir", default=".", help="Directory for per-case and summary CSV files")
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--num_slices", type=int, default=10)
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--fid_feature_mode", choices=["raw2048", "pca64"], default="raw2048")
    parser.add_argument("--fid_point_estimate", choices=["point", "median"], default="point")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = create_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_tag or Path(args.pred_dir).name
    per_case_csv = output_dir / f"per_case_registered_{model_tag}.csv"
    summary_csv = output_dir / f"metrics_registered_{model_tag}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    extractor = MedicalResNetFeatureExtractor().to(device)

    gt_subdirs = sorted([d for d in os.listdir(args.gt_dir) if os.path.isdir(os.path.join(args.gt_dir, d))])
    pred_subdirs = sorted([d for d in os.listdir(args.pred_dir) if os.path.isdir(os.path.join(args.pred_dir, d))])
    common = sorted(set(gt_subdirs).intersection(pred_subdirs))
    if common:
        pairs = [(case, case) for case in common]
    elif len(gt_subdirs) == len(pred_subdirs):
        pairs = list(zip(gt_subdirs, pred_subdirs))
    else:
        pairs = []

    exclude = set(args.exclude)
    pairs = [(gt_case, pred_case) for gt_case, pred_case in pairs if gt_case not in exclude]

    print(f"Device: {device}")
    print(f"Evaluation pairs: {len(pairs)}")
    case_rows = []
    real_case_features = []
    fake_case_features = []
    registration_failures = 0

    for gt_case, pred_case in tqdm(pairs, desc="Registering and evaluating"):
        gt_case_dir = os.path.join(args.gt_dir, gt_case)
        pred_case_dir = os.path.join(args.pred_dir, pred_case)
        gt_path = find_gt_file(gt_case_dir)
        pred_path = find_pred_file(pred_case_dir)
        if gt_path is None or pred_path is None:
            print(f"Skipping {gt_case}: missing GT or prediction")
            continue

        try:
            gt_raw = to_dhw(nib.load(gt_path).get_fdata(dtype=np.float32))
            gt_aligned = interpolate_to_target(gt_raw)
            pred_raw = to_dhw(nib.load(pred_path).get_fdata(dtype=np.float32))
            pred_aligned = interpolate_to_target(pred_raw)

            try:
                pred_registered = ants_rigid_register(gt_aligned, pred_aligned)
            except Exception as error:
                registration_failures += 1
                pred_registered = pred_aligned
                print(f"{gt_case}: registration failed, using unregistered prediction: {error}")

            gt_norm = normalize_to_0_1(gt_aligned)
            pred_norm = normalize_to_0_1(pred_registered)
            result = compute_case_metrics(
                gt_norm,
                pred_norm,
                lpips_model,
                extractor,
                device,
                num_slices=args.num_slices,
            )

            case_rows.append({
                "subject": gt_case,
                "LPIPS": result["LPIPS"],
                "ROI_MAE": result["ROI_MAE"],
                "MSE": result["MSE"],
                "dice": result["dice"],
            })
            real_case_features.append(result["gt_features"])
            fake_case_features.append(result["pred_features"])
        except Exception as error:
            print(f"Failed to process {gt_case}: {error}")

    if not case_rows:
        raise RuntimeError("No cases were evaluated successfully")

    per_case = pd.DataFrame(case_rows)
    per_case.to_csv(per_case_csv, index=False)

    lpips_mean, lpips_lo, lpips_hi = bootstrap_mean_ci(per_case["LPIPS"], args.bootstrap, args.seed)
    roi_mean, roi_lo, roi_hi = bootstrap_mean_ci(per_case["ROI_MAE"], args.bootstrap, args.seed + 1)
    mse_mean, mse_lo, mse_hi = bootstrap_mean_ci(per_case["MSE"], args.bootstrap, args.seed + 2)
    dice_mean, dice_lo, dice_hi = bootstrap_mean_ci(per_case["dice"], args.bootstrap, args.seed + 3)

    (fid_median, fid_lo, fid_hi), (mmd_median, mmd_lo, mmd_hi) = bootstrap_feature_distances(
        real_case_features,
        fake_case_features,
        num_bootstrap=args.bootstrap,
        seed=args.seed,
        pca_components=args.pca_components,
        feature_mode=args.fid_feature_mode,
        point_estimate=args.fid_point_estimate,
    )

    summary = {
        "model": model_tag,
        "LPIPS": lpips_mean,
        "LPIPS_CI_lo": lpips_lo,
        "LPIPS_CI_hi": lpips_hi,
        "FID": fid_median,
        "FID_CI_lo": fid_lo,
        "FID_CI_hi": fid_hi,
        "MMD": mmd_median,
        "MMD_CI_lo": mmd_lo,
        "MMD_CI_hi": mmd_hi,
        "ROI_MAE": roi_mean,
        "ROI_MAE_CI_lo": roi_lo,
        "ROI_MAE_CI_hi": roi_hi,
        "MSE": mse_mean,
        "MSE_CI_lo": mse_lo,
        "MSE_CI_hi": mse_hi,
        "Dice": dice_mean,
        "Dice_CI_lo": dice_lo,
        "Dice_CI_hi": dice_hi,
        "n": len(per_case),
        "registration_failures": registration_failures,
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Per-case CSV: {per_case_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()







