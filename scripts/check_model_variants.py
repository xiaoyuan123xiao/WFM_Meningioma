"""Check WFM parameter counts and optional forward shapes.

Run this in the same environment used for training:

    python scripts/check_model_variants.py
    python scripts/check_model_variants.py --forward

The official WFM implementation and the meningioma experiments use additive
condition injection (`use_scale_shift_norm=False`). This script documents the
parameter-count difference for that setting versus an AdaGN exploratory variant.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from guided_diffusion.script_util import create_model
from guided_diffusion.wfm import DiscrepancyPredictor


def build(in_channels, use_scale_shift_norm):
    return create_model(
        image_size=512,
        num_channels=64,
        num_res_blocks=2,
        channel_mult="1,2,2,4,4",
        learn_sigma=False,
        class_cond=True,
        use_checkpoint=False,
        attention_resolutions="",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=0.0,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        dims=3,
        num_groups=32,
        in_channels=in_channels,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
    )


def count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()

    for use_discrepancy in (False, True):
        in_channels = 32 + int(use_discrepancy)
        for scale_shift in (False, True):
            model = build(in_channels, scale_shift)
            model.eval()
            total = count(model)
            line = (
                f"DAGCN={use_discrepancy} AdaGN={scale_shift} "
                f"in_channels={in_channels} model_params={total:,}"
            )
            if use_discrepancy:
                predictor = DiscrepancyPredictor()
                line += f" predictor_params={count(predictor):,} total={total + count(predictor):,}"
            print(line)

            if args.forward:
                x = torch.randn(1, in_channels, 16, 32, 32)
                t = torch.zeros(1)
                y = torch.ones(1, dtype=torch.long)
                with torch.no_grad():
                    out = model(x, t, y=y)
                print(f"  forward: input={tuple(x.shape)} output={tuple(out.shape)}")


if __name__ == "__main__":
    main()

