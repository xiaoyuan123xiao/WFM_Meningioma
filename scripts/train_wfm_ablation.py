"""Train the WFM meningioma T1c ablation models.

Example:
    python scripts/train_wfm_ablation.py \
      --variant dagcn --data_dir /data/meningioma/train \
      --max_iterations 500 --output_dir runs/dagcn_500
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from guided_diffusion.ablation_losses import ssim_loss
from guided_diffusion.anatomy_loss import AnatomyConsistencyLoss
from guided_diffusion.meningioma_loader import MeningiomaMRIDataset
from guided_diffusion.script_util import create_model
from guided_diffusion.wfm import HF_CHANNELS, WaveletFlowMatching


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_variant(args):
    presets = {
        # Manuscript cumulative ablation design.
        "original": dict(use_high_freq=False, use_discrepancy=False, use_anatomy=False),
        "highfreq_pure": dict(use_high_freq=True, use_discrepancy=False, use_anatomy=False),
        "dagcn": dict(use_high_freq=True, use_discrepancy=True, use_anatomy=False),
        "anatomy": dict(use_high_freq=True, use_discrepancy=True, use_anatomy=True),
        "combined": dict(use_high_freq=True, use_discrepancy=True, use_anatomy=True),
        # Supplementary independent ablations.
        "fm_diff": dict(use_high_freq=False, use_discrepancy=True, use_anatomy=False),
        "fm_anat": dict(use_high_freq=False, use_discrepancy=False, use_anatomy=True),
        "fm_hf_anat": dict(use_high_freq=True, use_discrepancy=False, use_anatomy=True),
        # Result-table aliases used by the supplementary ablation CSVs.
        "diff": dict(use_high_freq=False, use_discrepancy=True, use_anatomy=False),
        "anat": dict(use_high_freq=False, use_discrepancy=False, use_anatomy=True),
        "hf_anat": dict(use_high_freq=True, use_discrepancy=False, use_anatomy=True),
    }
    preset = presets[args.variant]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.use_high_freq = bool(args.use_high_freq)
    args.use_discrepancy = bool(args.use_discrepancy)
    args.use_anatomy = bool(args.use_anatomy)
    return args

def build_model(args):
    in_channels = 32 + (1 if args.use_discrepancy else 0)
    return create_model(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=args.channel_mult,
        learn_sigma=False,
        class_cond=True,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=False,
        use_new_attention_order=args.use_new_attention_order,
        dims=3,
        num_groups=args.num_groups,
        in_channels=in_channels,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
    )


def build_wfm(args):
    return WaveletFlowMatching(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        mode="i2i",
        use_discrepancy=args.use_discrepancy,
        discrepancy_target=args.discrepancy_target,
        discrepancy_input=args.discrepancy_input,
        source_prior_mode=args.source_prior_mode,
    )

def compute_ablation_losses(args, model, wfm, batch, anatomy_loss, device):
    batch_size = batch["t1c"].shape[0]
    t = torch.randint(0, wfm.num_timesteps, (batch_size,), device=device)
    terms, model_output, pred_img = wfm.training_losses(
        model=model,
        x_start=batch,
        t=t,
        contr=args.target,
    )
    fm_loss = terms["mse_wav"].mean()
    loss = fm_loss
    logs = {"loss/fm": float(fm_loss.detach().cpu())}
    if args.use_high_freq:
    parser.add_argument("--high_freq_time", choices=["zero", "current"], default="zero")
            zero_target = wfm.predict_target_at_zero(model, batch, contr=args.target)
            hf_pred = zero_target if args.high_freq_prediction == "pred_target" else zero_target - terms["source_dwt"]
        else:
            hf_pred = terms["pred_target_dwt"] if args.high_freq_prediction == "pred_target" else model_output
        hf_loss = F.mse_loss(hf_pred[:, HF_CHANNELS], terms["target_dwt"][:, HF_CHANNELS])
        loss = loss + args.high_freq_weight * hf_loss
        logs["loss/high_freq"] = float(hf_loss.detach().cpu())

    if args.use_discrepancy:
        disc_loss = F.mse_loss(terms["discrepancy_map"], terms["discrepancy_target"])
        loss = loss + args.discrepancy_weight * disc_loss
        logs["loss/discrepancy"] = float(disc_loss.detach().cpu())

    if args.use_anatomy and anatomy_loss is not None:
        anat_loss = anatomy_loss(pred_img, batch[args.target])
        loss = loss + args.anatomy_weight * anat_loss
        logs["loss/anatomy"] = float(anat_loss.detach().cpu())

    if args.ssim_weight > 0:
        ssim_value = ssim_loss(pred_img, batch[args.target])
        loss = loss + args.ssim_weight * ssim_value
        logs["loss/ssim"] = float(ssim_value.detach().cpu())

    logs["loss/total"] = float(loss.detach().cpu())
    return loss, logs

def save_checkpoint(path, model, wfm, optimizer, scheduler, args, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "wfm": wfm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
        },
        str(path),
    )


def main():
    args = apply_variant(create_argparser().parse_args())
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    wfm = build_wfm(args).to(device)
    anatomy_loss = AnatomyConsistencyLoss(device=device, arch=args.anatomy_arch) if args.use_anatomy else None

    train_set = MeningiomaMRIDataset(
        args.data_dir,
        target_shape=(args.target_depth, args.target_size, args.target_size),
        augment=True,
        rotation_deg=args.rotation_deg,
        strict=True,
        exclude_cases=args.exclude_cases,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; increase the dataset size or lower batch_size")

    parameters = list(model.parameters()) + list(wfm.parameters())
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iterations, eta_min=args.eta_min)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if args.tensorboard else None

    iterator = iter(train_loader)
    start_time = time.time()
    for step in range(1, args.max_iterations + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            loss, logs = compute_ablation_losses(args, model, wfm, batch, anatomy_loss, device)

        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss; skipping update")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.log_interval == 0 or step == 1 or step == args.max_iterations:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            message = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
            print(f"step {step}/{args.max_iterations} lr={lr:.3e} {message} elapsed={elapsed:.1f}s")
            if writer is not None:
                writer.add_scalar("train/lr", lr, step)
                for key, value in logs.items():
                    writer.add_scalar(key, value, step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(output_dir / "checkpoints" / f"step_{step:06d}.pt", model, wfm, optimizer, scheduler, args, step)

    save_checkpoint(output_dir / "checkpoints" / f"step_{args.max_iterations:06d}.pt", model, wfm, optimizer, scheduler, args, args.max_iterations)
    if writer is not None:
        writer.close()
    print(f"Training complete. Checkpoint: {output_dir / 'checkpoints' / f'step_{args.max_iterations:06d}.pt'}")

def create_argparser():
    parser = argparse.ArgumentParser(description="WFM meningioma T1c ablation training")
    parser.add_argument("--variant", choices=["original", "highfreq_pure", "dagcn", "anatomy", "combined", "fm_diff", "fm_anat", "fm_hf_anat", "diff", "anat", "hf_anat"], default="original")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target", default="t1c")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no_tensorboard", action="store_false", dest="tensorboard")

    parser.add_argument("--use_high_freq", action="store_true", default=None)
    parser.add_argument("--no_use_high_freq", action="store_false", dest="use_high_freq", default=None)
    parser.add_argument("--high_freq_weight", type=float, default=0.1)
    parser.add_argument("--high_freq_prediction", choices=["pred_target", "velocity"], default="pred_target")
    parser.add_argument("--high_freq_time", choices=["zero", "current"], default="zero")
    parser.add_argument("--use_discrepancy", action="store_true", default=None)
    parser.add_argument("--no_use_discrepancy", action="store_false", dest="use_discrepancy", default=None)
    parser.add_argument("--discrepancy_weight", type=float, default=0.1)
    parser.add_argument("--discrepancy_target", choices=["hf_mean", "lll_abs", "image_diff"], default="hf_mean")
    parser.add_argument("--discrepancy_input", choices=["predicted", "target"], default="predicted")
    parser.add_argument("--use_anatomy", action="store_true", default=None)
    parser.add_argument("--no_use_anatomy", action="store_false", dest="use_anatomy", default=None)
    parser.add_argument("--anatomy_weight", type=float, default=0.05)
    parser.add_argument("--anatomy_arch", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--ssim_weight", type=float, default=0.0)
    parser.add_argument("--source_prior_mode", choices=["mean", "t1n_anchor"], default="mean")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--target_depth", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--rotation_deg", type=float, default=5.0)
    parser.add_argument("--exclude_cases", nargs="*", default=[])
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", default="1,2,2,4,4")
    parser.add_argument("--attention_resolutions", default="")
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_head_channels", type=int, default=-1)
    parser.add_argument("--num_heads_upsample", type=int, default=-1)
    parser.add_argument("--num_groups", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--use_scale_shift_norm", action="store_true", default=False)
    parser.add_argument("--no_use_scale_shift_norm", action="store_false", dest="use_scale_shift_norm")
    parser.add_argument("--resblock_updown", action="store_true", default=True)
    parser.add_argument("--use_new_attention_order", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=0.5)
    return parser



if __name__ == "__main__":
    main()















