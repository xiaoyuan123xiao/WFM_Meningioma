# Reproducibility Notes

## Conditioning implementation

The official WFM training script uses:

```python
use_scale_shift_norm=False
```

In this setting the time and modality embeddings are injected additively into
the residual block rather than used for AdaGN/FiLM scale-and-shift modulation.
The reconstructed manuscript experiments follow that setting.

`--use_scale_shift_norm` is available only for exploratory AdaGN variants and
is not the default for the reported ablation models.

## WFM paper and released code

The WFM paper describes conditional modulation through adaptive normalization,
while the official released training script sets `use_scale_shift_norm=False`.
This repository follows the released training code and preserves the distinction
in the documentation.

## DAGCN variants

Two DAGCN behaviors are supported:

- manuscript reconstruction: seven-band mean absolute discrepancy target,
  predicted map used as the training-time 33rd input channel
- recovered release variant: LLL absolute difference target, detached
  ground-truth map used as the 33rd input channel

Use:

```text
--discrepancy_target hf_mean --discrepancy_input predicted
--discrepancy_target lll_abs --discrepancy_input target
```

## High-frequency loss variants

The default manuscript behavior uses the predicted target wavelet coefficients:

```text
x_source + v(x_source, c, 0, y)
```

The historical experimental implementation can be selected with
`--high_freq_prediction velocity`.

## Anatomy-loss variants

The default `paper` architecture uses:

```text
Conv3d -> InstanceNorm3d -> ReLU -> MaxPool3d
```

The recovered historical architecture is available with
`--anatomy_arch legacy`.

## Data reproducibility

The original patient datasets are not redistributed. Reproducing the numerical
results requires the approved, de-identified datasets and the original split
manifest. The code can be tested on any dataset arranged in the documented
directory format.
