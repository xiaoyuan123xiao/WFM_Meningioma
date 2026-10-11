"""
配准后评估脚本
- 真实图像和合成图像都插值到 32 层
- 用 ANTs 刚性配准，将合成图像对齐到真实图像
- 在配准后的图像上计算 LPIPS、FID、MMD、ROI-MAE、MSE
- 逐例保存，用于统计检验
"""

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
import lpips
import ants
from tqdm import tqdm
import argparse
import warnings
from scipy.ndimage import binary_fill_holes, label

warnings.filterwarnings("ignore")

import torch.nn as nn
import torchvision.models as models


TARGET_SHAPE = (32, 512, 512)  # (D, H, W)


# ========== 特征提取器 ==========
class MedicalResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
            features = features.view(features.size(0), -1)
        return features


def normalize_to_0_1(img):
    low = np.quantile(img, 0.001)
    high = np.quantile(img, 0.999)
    if high - low < 1e-8:
        return np.zeros_like(img)
    return np.clip((np.clip(img, low, high) - low) / (high - low), 0.0, 1.0)


def to_dhw(volume):
    """任意 (H,W,D) 或 (D,H,W) -> (D,H,W)"""
    if volume.shape[0] <= volume.shape[1] and volume.shape[0] <= volume.shape[2]:
        return volume
    return np.transpose(volume, (2, 0, 1))


def interpolate_to_target(volume_dhw, target_shape=TARGET_SHAPE):
    if volume_dhw.shape == target_shape:
        return volume_dhw
    t = torch.from_numpy(volume_dhw).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def ants_rigid_register(fixed_dhw, moving_dhw):
    """
    用 ANTs 做刚性配准。
    - fixed_dhw, moving_dhw: (D, H, W) numpy 数组
    - 返回：配准后的 moving 图像 (D, H, W)
    """
    fixed_hwd = np.transpose(fixed_dhw, (1, 2, 0))
    moving_hwd = np.transpose(moving_dhw, (1, 2, 0))

    fixed = ants.from_numpy(fixed_hwd)
    moving = ants.from_numpy(moving_hwd)

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='Rigid',
        verbose=False
    )
    warped_hwd = reg['warpedmovout'].numpy()
    return np.transpose(warped_hwd, (2, 0, 1))


def get_brain_mask(img, threshold=0.1):
    mask = img > threshold
    mask = binary_fill_holes(mask)
    labeled, num = label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    if len(sizes) <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labeled == (np.argmax(sizes[1:]) + 1)


def compute_metrics(gt_norm, pred_norm, lpips_model, extractor, device, num_slices=10):
    """在配准后的图像上计算指标"""
    mask = get_brain_mask(gt_norm, threshold=0.1)
    mask_pred = get_brain_mask(pred_norm, threshold=0.1)

    inter = np.logical_and(mask, mask_pred).sum()
    union = mask.sum() + mask_pred.sum()
    dice = 2.0 * inter / union if union > 0 else 0.0

    roi_mae_val = float(np.mean(np.abs(gt_norm[mask] - pred_norm[mask]))) \
        if mask.sum() > 0 else np.nan

    mse_val = float(np.mean((gt_norm - pred_norm) ** 2))

    mid = gt_norm.shape[0] // 2
    gt_slice = gt_norm[mid]
    pred_slice = pred_norm[mid]
    gt_t = torch.from_numpy(gt_slice).float().unsqueeze(0).unsqueeze(0) * 2 - 1
    pred_t = torch.from_numpy(pred_slice).float().unsqueeze(0).unsqueeze(0) * 2 - 1
    gt_t = gt_t.repeat(1, 3, 1, 1).to(device)
    pred_t = pred_t.repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        lpips_val = float(lpips_model(gt_t, pred_t).item())

    feats_gt = []
    feats_pred = []
    indices = np.linspace(0, gt_norm.shape[0] - 1, num_slices, dtype=int)
    for i in indices:
        s_gt = gt_norm[i]
        s_pred = pred_norm[i]
        t_gt = torch.from_numpy(s_gt).float().unsqueeze(0).unsqueeze(0) * 2 - 1
        t_pred = torch.from_numpy(s_pred).float().unsqueeze(0).unsqueeze(0) * 2 - 1
        t_gt = t_gt.repeat(1, 3, 1, 1).to(device)
        t_pred = t_pred.repeat(1, 3, 1, 1).to(device)
        feats_gt.append(extractor(t_gt).cpu().numpy().flatten())
        feats_pred.append(extractor(t_pred).cpu().numpy().flatten())

    return {
        'LPIPS': lpips_val,
        'ROI_MAE': roi_mae_val,
        'MSE': mse_val,
        'dice': dice,
        'gt_features': np.array(feats_gt),
        'pred_features': np.array(feats_pred),
    }


def compute_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    eps = 1e-6
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
    diff = mu1 - mu2
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_mmd(real_features, fake_features, sigma=None):
    X = real_features
    Y = fake_features
    X_norm = np.sum(X ** 2, axis=1, keepdims=True)
    Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
    XX = np.maximum(X_norm + X_norm.T - 2 * X @ X.T, 0)
    YY = np.maximum(Y_norm + Y_norm.T - 2 * Y @ Y.T, 0)
    XY = np.maximum(X_norm + Y_norm.T - 2 * X @ Y.T, 0)
    if sigma is None:
        all_dists = np.concatenate([XX.flatten(), YY.flatten(), XY.flatten()])
        sigma = np.median(np.sqrt(all_dists)) + 1e-8
    K_XX = np.exp(-XX / (2 * sigma ** 2))
    K_YY = np.exp(-YY / (2 * sigma ** 2))
    K_XY = np.exp(-XY / (2 * sigma ** 2))
    return float(np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True)
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--output_csv', default='metrics_registered.csv')
    parser.add_argument('--model_tag', default=None)
    parser.add_argument('--num_slices', type=int, default=10)
    parser.add_argument('--exclude', nargs='*', default=[])
    args = parser.parse_args()

    if args.model_tag is None:
        base = os.path.basename(args.output_csv)
        model_tag = base.replace('metrics_registered_', '').replace('.csv', '')
        if model_tag == base.replace('.csv', ''):
            model_tag = base.replace('metrics_', '').replace('.csv', '')
    else:
        model_tag = args.model_tag
    per_case_csv = f'per_case_registered_{model_tag}.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
    extractor = MedicalResNetFeatureExtractor().to(device)

    gt_subdirs = sorted([d for d in os.listdir(args.gt_dir)
                         if os.path.isdir(os.path.join(args.gt_dir, d))])
    pred_subdirs = sorted([d for d in os.listdir(args.pred_dir)
                           if os.path.isdir(os.path.join(args.pred_dir, d))])

    if all(d.startswith('sample_') for d in pred_subdirs):
        n = min(len(gt_subdirs), len(pred_subdirs))
        pairs = [(gt_subdirs[i], pred_subdirs[i]) for i in range(n)]
    else:
        common = set(gt_subdirs).intersection(set(pred_subdirs))
        pairs = [(s, s) for s in common]

    if args.exclude:
        ex = set(args.exclude)
        pairs = [(g, p) for g, p in pairs if g not in ex]
        print(f"排除病例后剩余: {len(pairs)} 对")

    case_results = []
    all_gt_feats = []
    all_pred_feats = []
    reg_failures = 0

    for gt_subj, pred_subj in tqdm(pairs, desc="配准+评估中"):
        gt_dir = os.path.join(args.gt_dir, gt_subj)
        pred_dir = os.path.join(args.pred_dir, pred_subj)

        gt_files = glob.glob(os.path.join(gt_dir, '*t1c*.nii.gz'))
        gt_files = [f for f in gt_files
                    if 'sample' not in os.path.basename(f).lower()
                    and 'target' not in os.path.basename(f).lower()]
        if not gt_files:
            gt_files = [f for f in glob.glob(os.path.join(gt_dir, '*.nii.gz'))
                        if 't1c' in os.path.basename(f).lower()]
        if not gt_files:
            continue
        gt_path = gt_files[0]
        pred_path = os.path.join(pred_dir, 'sample_t1c.nii.gz')
        if not os.path.exists(pred_path):
            continue

        try:
            gt_raw = to_dhw(nib.load(gt_path).get_fdata().astype(np.float32))
            gt_aligned = interpolate_to_target(gt_raw)
            pred_raw = to_dhw(nib.load(pred_path).get_fdata().astype(np.float32))
            pred_aligned = interpolate_to_target(pred_raw)

            try:
                pred_reg = ants_rigid_register(gt_aligned, pred_aligned)
            except Exception as e:
                print(f"  {gt_subj} 配准失败: {e}")
                reg_failures += 1
                pred_reg = pred_aligned

            gt_norm = normalize_to_0_1(gt_aligned)
            pred_norm = normalize_to_0_1(pred_reg)

            res = compute_metrics(gt_norm, pred_norm,
                                  lpips_model, extractor, device,
                                  num_slices=args.num_slices)

            case_results.append({
                'subject': gt_subj,
                'LPIPS': res['LPIPS'],
                'ROI_MAE': res['ROI_MAE'],
                'MSE': res['MSE'],
                'dice': res['dice'],
            })
            all_gt_feats.append(res['gt_features'])
            all_pred_feats.append(res['pred_features'])

        except Exception as e:
            print(f"处理 {gt_subj} 时出错: {e}")

    print(f"\n配准失败病例数: {reg_failures}")

    if not case_results:
        print("没有成功处理的样本。")
        return

    df = pd.DataFrame(case_results)
    df.to_csv(per_case_csv, index=False)
    print(f"逐例结果已保存到: {per_case_csv}")

    results = {
        'LPIPS': df['LPIPS'].mean(),
        'LPIPS_std': df['LPIPS'].std(),
        'ROI_MAE': df['ROI_MAE'].mean(),
        'ROI_MAE_std': df['ROI_MAE'].std(),
        'MSE': df['MSE'].mean(),
        'MSE_std': df['MSE'].std(),
        'dice_mean': df['dice'].mean(),
        'reg_failures': reg_failures,
        'n': len(df),
    }

    if all_gt_feats and all_pred_feats:
        real_feats = np.vstack(all_gt_feats)
        fake_feats = np.vstack(all_pred_feats)
        results['FID'] = compute_fid(real_feats, fake_feats)
        results['MMD'] = compute_mmd(real_feats, fake_feats)

    print("\n" + "=" * 60)
    print(f"配准后评估结果 [{model_tag}] (n={results['n']})")
    print("=" * 60)
    print(f"LPIPS:    {results['LPIPS']:.4f} ± {results['LPIPS_std']:.4f}")
    print(f"FID:      {results.get('FID', np.nan):.2f}")
    print(f"MMD:      {results.get('MMD', np.nan):.6f}")
    print(f"ROI-MAE:  {results['ROI_MAE']:.4f} ± {results['ROI_MAE_std']:.4f}")
    print(f"MSE:      {results['MSE']:.4f} ± {results['MSE_std']:.4f}")
    print(f"脑掩码 Dice: {results['dice_mean']:.4f}")
    print(f"配准失败数: {reg_failures}")
    print("=" * 60)

    pd.DataFrame([results]).to_csv(args.output_csv, index=False)
    print(f"汇总结果已保存到: {args.output_csv}")


if __name__ == '__main__':
    main()