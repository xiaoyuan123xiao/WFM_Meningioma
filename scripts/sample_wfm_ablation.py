"""Sample T1c volumes from a trained WFM ablation checkpoint."""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from guided_diffusion.meningioma_loader import MeningiomaMRIDataset
from train_wfm_ablation import apply_variant, build_model, build_wfm, create_argparser


def load_checkpoint(path, model, wfm, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        if "wfm" in checkpoint and checkpoint["wfm"]:
            wfm.load_state_dict(checkpoint["wfm"], strict=False)
    else:
        model.load_state_dict(checkpoint)
    step = checkpoint.get("step", "unknown") if isinstance(checkpoint, dict) else "unknown"
    return step


def to_nifti(array):
    # Archived dagcn_200 outputs are stored as (D, H, W) = (32, 512, 512).
    return array

def main():
    parser = create_argparser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sampling_steps", type=int, default=1)
    parser.add_argument("--solver", choices=["direct", "euler", "heun"], default="direct")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--save_target", action="store_true", default=True)
    parser.add_argument("--no_save_target", action="store_false", dest="save_target")
    args = apply_variant(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args).to(device)
    wfm = build_wfm(args).to(device)
    step = load_checkpoint(args.checkpoint, model, wfm, device)
    model.eval()
    wfm.eval()

    dataset = MeningiomaMRIDataset(
        args.data_dir,
        target_shape=(args.target_depth, args.target_size, args.target_size),
        augment=False,
        strict=True,
        exclude_cases=args.exclude_cases,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    records = []
    for index, batch in enumerate(tqdm(loader, desc="Sampling")):
        if args.num_samples >= 0 and index >= args.num_samples:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        case_id = batch["subj"][0] if isinstance(batch["subj"], (list, tuple)) else str(batch["subj"])
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            output_img, target = wfm.sample_for_evaluation(
                model,
                batch,
                contr=args.target,
                num_steps=args.sampling_steps,
                device=device,
                progress=False,
                clip_denoised=True,
                solver=args.solver,
            )

        output_np = to_nifti(output_img[0, 0].detach().cpu().numpy())
        nib.save(nib.Nifti1Image(output_np.astype(np.float32), np.eye(4)), str(case_dir / "sample_t1c.nii.gz"))
        if args.save_target:
            target_np = to_nifti(target[0, 0].detach().cpu().numpy())
            nib.save(nib.Nifti1Image(target_np.astype(np.float32), np.eye(4)), str(case_dir / "target_t1c.nii.gz"))
        records.append({"case": case_id, "path": str(case_dir / "sample_t1c.nii.gz")})

    (output_dir / "sampling_summary.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "step": step, "records": records}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()



