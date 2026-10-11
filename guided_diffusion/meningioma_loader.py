"""Dataset loader for the meningioma WFM ablation experiments."""

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def normalize_percentile(image, lower=0.001, upper=0.999):
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, copy=False)
    lo = np.quantile(image, lower)
    hi = np.quantile(image, upper)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def match_modality(path, modality):
    name = path.name.lower()
    return (name.endswith(f"-{modality}.nii.gz") or
            name.endswith(f"_{modality}.nii.gz") or
            f"-{modality}_" in name or f"_{modality}_" in name)

class MeningiomaMRIDataset(Dataset):
    def __init__(self, root, target_shape=(32, 512, 512), augment=False, rotation_deg=5.0, strict=True, exclude_cases=()):
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.target_shape = tuple(int(v) for v in target_shape)
        self.augment = bool(augment)
        self.rotation_deg = float(rotation_deg)
        self.exclude_cases = set(exclude_cases)
        self.cases = []
        for case_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            if case_dir.name in self.exclude_cases:
                continue
            files = [p for p in case_dir.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))]
            mapping = {}
            for modality in MODALITIES:
                matches = sorted([p for p in files if match_modality(p, modality)])
                if matches:
                    mapping[modality] = matches[0]
            if len(mapping) == len(MODALITIES):
                self.cases.append((case_dir.name, mapping))
            elif strict:
                missing = sorted(set(MODALITIES) - set(mapping))
                raise ValueError(f"{case_dir} is missing modalities: {missing}")
        if not self.cases:
            raise ValueError(f"No complete cases found under {self.root}")

    def __len__(self):
        return len(self.cases)

    def _load_volume(self, path):
        image = nib.load(str(path))
        array = normalize_percentile(image.get_fdata(dtype=np.float32))
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=self.target_shape, mode="trilinear", align_corners=False)
        return tensor[0]

    def _augment_stack(self, stack):
        if random.random() < 0.5:
            stack = torch.flip(stack, dims=(2,))
        if random.random() < 0.5:
            stack = torch.flip(stack, dims=(3,))
        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            radians = angle * np.pi / 180.0
            cos_a, sin_a = float(np.cos(radians)), float(np.sin(radians))
            theta = stack.new_tensor([[[cos_a, -sin_a, 0.0, 0.0],
                                       [sin_a, cos_a, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0]]])
            grid = F.affine_grid(theta, list(stack.shape), align_corners=False)
            stack = F.grid_sample(stack, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return stack

    def __getitem__(self, index):
        case_id, mapping = self.cases[index]
        stack = torch.cat([self._load_volume(mapping[m]) for m in MODALITIES], dim=0)
        if self.augment:
            stack = self._augment_stack(stack)
        t1n, t1c, t2w, t2f = stack.unbind(0)
        return {
            "t1n": t1n.unsqueeze(0).float(),
            "t1c": t1c.unsqueeze(0).float(),
            "t2w": t2w.unsqueeze(0).float(),
            "t2f": t2f.unsqueeze(0).float(),
            "subj": case_id,
            "missing": "none",
        }


