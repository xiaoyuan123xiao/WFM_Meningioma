"""
Wavelet Flow Matching for meningioma non-contrast MRI to T1c synthesis.

This is a clean reconstruction of the WFM ablation implementation described in
the accompanying manuscript.  The original WFM code path is preserved, while
the three ablation mechanisms are exposed through explicit switches:

  * high-frequency wavelet loss
  * DAGCN discrepancy-map guidance
  * multi-scale anatomical consistency loss

The discrepancy-map target and input semantics are configurable because the
released experimental code and the final manuscript describe different
variants:

  discrepancy_target='hf_mean'   # manuscript: mean |HF(target)-HF(source)|
  discrepancy_target='lll_abs'   # recovered release code: |LLL(target)-LLL(source)|
  discrepancy_target='image_diff' # source image mean vs target image

  discrepancy_input='predicted'  # manuscript: predictor output is the 33rd channel
  discrepancy_input='target'     # recovered release code: detached GT map at train time
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import mean_flat
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt = DWT_3D("haar")
idwt = IDWT_3D("haar")

MODALITY_TO_CLASS = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
CLASS_TO_MODALITY = {0: "t1n", 1: "t1c", 2: "t2w", 3: "t2f"}
ALL_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
HF_CHANNELS = tuple(range(1, 8))


def apply_dwt(img):
    """Return the eight 3D Haar subbands concatenated along channels."""
    lll, llh, lhl, lhh, hll, hlh, hhl, hhh = dwt(img)
    return th.cat([lll / 3.0, llh, lhl, lhh, hll, hlh, hhl, hhh], dim=1)


def apply_idwt(x):
    """Invert :func:`apply_dwt`."""
    b, _, h, w, d = x.size()
    return idwt(
        x[:, 0].view(b, 1, h, w, d) * 3.0,
        x[:, 1].view(b, 1, h, w, d),
        x[:, 2].view(b, 1, h, w, d),
        x[:, 3].view(b, 1, h, w, d),
        x[:, 4].view(b, 1, h, w, d),
        x[:, 5].view(b, 1, h, w, d),
        x[:, 6].view(b, 1, h, w, d),
        x[:, 7].view(b, 1, h, w, d),
    )


def build_source_prior(cond_1_dwt, cond_2_dwt, cond_3_dwt, contr, mode="mean", weights=None):
    """Construct the informed source used by WFM."""
    if mode == "mean":
        return (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0
    if mode == "t1n_anchor" and contr == "t1c":
        w = (0.6, 0.2, 0.2) if weights is None else tuple(float(v) for v in weights)
        if len(w) != 3 or abs(sum(w) - 1.0) > 1e-4:
            raise ValueError(f"source_prior_weights must have three values summing to 1, got {w}")
        return w[0] * cond_1_dwt + w[1] * cond_2_dwt + w[2] * cond_3_dwt
    if mode != "t1n_anchor":
        raise ValueError(f"Unknown source prior mode: {mode}")
    return (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0


class DiscrepancyPredictor(nn.Module):
    """Lightweight 3D CNN used by DAGCN.

    Input:  [B, 24, D/2, H/2, W/2]
    Output: [B, 1,  D/2, H/2, W/2]
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, cond_dwt):
        return self.net(cond_dwt)


class WaveletFlowMatching(nn.Module):
    """Flow matching in the 3D Haar wavelet domain."""

    def __init__(
        self,
        num_timesteps=1000,
        sigma_max=0.5,
        mode="i2i",
        use_discrepancy=False,
        discrepancy_target="hf_mean",
        discrepancy_input="predicted",
        source_prior_mode="mean",
        source_prior_weights=None,
    ):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        self.sigma_max = float(sigma_max)
        self.mode = mode
        self.use_discrepancy = bool(use_discrepancy)
        self.discrepancy_target = discrepancy_target
        self.discrepancy_input = discrepancy_input
        self.source_prior_mode = source_prior_mode
        self.source_prior_weights = None if source_prior_weights is None else tuple(source_prior_weights)

        if self.discrepancy_input not in {"predicted", "target"}:
            raise ValueError("discrepancy_input must be 'predicted' or 'target'")
        if self.discrepancy_target not in {"hf_mean", "lll_abs", "image_diff"}:
            raise ValueError("discrepancy_target must be 'hf_mean', 'lll_abs', or 'image_diff'")

        if self.use_discrepancy:
            self.discrepancy_predictor = DiscrepancyPredictor()
        else:
            self.discrepancy_predictor = None

        t = th.linspace(0.0, 1.0, self.num_timesteps)
        self.register_buffer("sigmas", self.sigma_max * th.sqrt(t * (1.0 - t) + 1e-8), persistent=False)

    def get_sigma(self, t):
        return self.sigma_max * th.sqrt(t * (1.0 - t) + 1e-8)

    def q_sample(self, x_source, x_target, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_target)
        t_broadcast = t.view(-1, 1, 1, 1, 1)
        x_t = (1.0 - t_broadcast) * x_source + t_broadcast * x_target
        sigma_t = self.get_sigma(t).view(-1, 1, 1, 1, 1)
        return x_t + sigma_t * noise

    def _scale_timesteps(self, t):
        return t * self.num_timesteps

    def _get_target_and_conditions(self, x_start, contr):
        if isinstance(contr, int):
            class_idx = int(contr)
            contr = CLASS_TO_MODALITY[class_idx]
        else:
            class_idx = MODALITY_TO_CLASS[contr]
        target = x_start[contr]
        conditions = [x_start[m] for m in ALL_MODALITIES if m != contr]
        return target, conditions, class_idx

    def _prepare_condition(self, conditions, contr):
        cond_1, cond_2, cond_3 = conditions
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)
        source_dwt = build_source_prior(
            cond_1_dwt,
            cond_2_dwt,
            cond_3_dwt,
            contr,
            mode=self.source_prior_mode,
            weights=self.source_prior_weights,
        )
        return cond_dwt, source_dwt, (cond_1_dwt, cond_2_dwt, cond_3_dwt)

    def compute_discrepancy_target(self, target_dwt, source_dwt, target_img, cond_imgs):
        """Create the DAGCN supervision target."""
        if self.discrepancy_target == "lll_abs":
            return th.abs(target_dwt[:, 0:1] - source_dwt[:, 0:1])
        if self.discrepancy_target == "hf_mean":
            return th.abs(target_dwt[:, HF_CHANNELS] - source_dwt[:, HF_CHANNELS]).mean(dim=1, keepdim=True)
        if self.discrepancy_target == "image_diff":
            source_spatial = (cond_imgs[0] + cond_imgs[1] + cond_imgs[2]) / 3.0
            true_diff = th.abs(target_img - source_spatial)
            return F.interpolate(
                true_diff,
                size=target_dwt.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        raise AssertionError(self.discrepancy_target)

    def _compose_model_input(self, x_t, cond_dwt, discrepancy_map=None):
        parts = [x_t, cond_dwt]
        if self.use_discrepancy and discrepancy_map is not None:
            parts.append(discrepancy_map)
        return th.cat(parts, dim=1)

    def predict_target_at_zero(self, model, x_start, contr="t1c", model_kwargs=None):
        """Manuscript high-frequency target: x_source + v(x_source, c, t=0, y)."""
        if model_kwargs is None:
            model_kwargs = {}
        target, conditions, class_idx = self._get_target_and_conditions(x_start, contr)
        cond_dwt, source_dwt, _ = self._prepare_condition(conditions, contr)
        discrepancy_input = None
        if self.use_discrepancy:
            discrepancy_pred = self.discrepancy_predictor(cond_dwt)
            if self.discrepancy_input == "target":
                discrepancy_input = self.compute_discrepancy_target(
                    target_dwt=apply_dwt(target),
                    source_dwt=source_dwt,
                    target_img=target,
                    cond_imgs=conditions,
                ).detach()
            else:
                discrepancy_input = discrepancy_pred
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=target.device, dtype=th.long)
        t_zero = th.zeros(batch_size, device=target.device, dtype=th.float32)
        x_cond = self._compose_model_input(source_dwt, cond_dwt, discrepancy_input)
        velocity = model(x_cond, t_zero, y=y, **model_kwargs)
        return source_dwt + velocity
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, mode="i2i", contr="t1c"):
        """Return flow-matching terms and auxiliary tensors for the train loop."""
        del mode  # kept for compatibility with the original interface
        if model_kwargs is None:
            model_kwargs = {}

        target, conditions, class_idx = self._get_target_and_conditions(x_start, contr)
        cond_dwt, source_dwt, cond_dwt_parts = self._prepare_condition(conditions, contr)
        target_dwt = apply_dwt(target)

        discrepancy_map = None
        discrepancy_target = None
        model_discrepancy = None
        if self.use_discrepancy:
            discrepancy_map = self.discrepancy_predictor(cond_dwt)
            discrepancy_target = self.compute_discrepancy_target(
                target_dwt=target_dwt,
                source_dwt=source_dwt,
                target_img=target,
                cond_imgs=conditions,
            )
            model_discrepancy = (
                discrepancy_map if self.discrepancy_input == "predicted" else discrepancy_target.detach()
            )

        t_continuous = t.float() / self.num_timesteps
        if noise is None:
            noise = th.randn_like(target)
        noise_dwt = apply_dwt(noise)
        x_t = self.q_sample(source_dwt, target_dwt, t_continuous, noise=noise_dwt)
        x_t_cond = self._compose_model_input(x_t, cond_dwt, model_discrepancy)

        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=target.device, dtype=th.long)
        t_scaled = self._scale_timesteps(t_continuous)
        model_output = model(x_t_cond, t_scaled, y=y, **model_kwargs)

        velocity = target_dwt - source_dwt
        mse_per_channel = mean_flat((velocity - model_output) ** 2)
        loss = th.mean(mse_per_channel, dim=0)

        pred_target_dwt = source_dwt + model_output
        model_output_idwt = apply_idwt(pred_target_dwt)

        terms = {
            "mse_wav": loss,
            "source_dwt": source_dwt,
            "target_dwt": target_dwt,
            "pred_target_dwt": pred_target_dwt,
            "discrepancy_map": discrepancy_map,
            "discrepancy_target": discrepancy_target,
            "x_t": x_t,
            "x_t_cond": x_t_cond,
            "model_output_idwt": model_output_idwt,
            "cond_dwt_parts": cond_dwt_parts,
        }
        return terms, model_output, model_output_idwt

    def p_sample_step_euler(
        self,
        model,
        x_t,
        t_curr,
        t_next,
        cond,
        source,
        y=None,
        model_kwargs=None,
        deterministic=True,
        discrepancy_map=None,
    ):
        del deterministic, source
        if model_kwargs is None:
            model_kwargs = {}
        b = x_t.shape[0]
        device = x_t.device
        t_model = th.full((b,), float(t_curr) * self.num_timesteps, device=device)
        x_cond = self._compose_model_input(x_t, cond, discrepancy_map)
        with th.no_grad():
            velocity = model(x_cond, t_model, y=y, **model_kwargs)
        return x_t + (float(t_next) - float(t_curr)) * velocity

    def p_sample_step_heun(
        self,
        model,
        x_t,
        t_curr,
        t_next,
        cond,
        source,
        y=None,
        model_kwargs=None,
        discrepancy_map=None,
    ):
        del source
        if model_kwargs is None:
            model_kwargs = {}
        b = x_t.shape[0]
        device = x_t.device
        dt = float(t_next) - float(t_curr)

        t_curr_tensor = th.full((b,), float(t_curr) * self.num_timesteps, device=device)
        x_cond = self._compose_model_input(x_t, cond, discrepancy_map)
        with th.no_grad():
            v1 = model(x_cond, t_curr_tensor, y=y, **model_kwargs)

        x_pred = x_t + dt * v1
        t_next_tensor = th.full((b,), float(t_next) * self.num_timesteps, device=device)
        x_pred_cond = self._compose_model_input(x_pred, cond, discrepancy_map)
        with th.no_grad():
            v2 = model(x_pred_cond, t_next_tensor, y=y, **model_kwargs)
        return x_t + 0.5 * dt * (v1 + v2)

    def p_sample_step(
        self,
        model,
        x_t,
        t_curr,
        t_next,
        cond,
        source,
        y=None,
        model_kwargs=None,
        clip_denoised=True,
        solver="euler",
        discrepancy_map=None,
    ):
        del clip_denoised
        if solver == "heun":
            return self.p_sample_step_heun(
                model, x_t, t_curr, t_next, cond, source, y, model_kwargs, discrepancy_map
            )
        return self.p_sample_step_euler(
            model, x_t, t_curr, t_next, cond, source, y, model_kwargs, True, discrepancy_map
        )

    def direct_prediction(self, model, cond, source, y=None, device=None, model_kwargs=None, discrepancy_map=None):
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        b = source.shape[0]
        t_model = th.zeros(b, device=device, dtype=th.float32)
        x_cond = self._compose_model_input(source, cond, discrepancy_map)
        with th.no_grad():
            velocity = model(x_cond, t_model, y=y, **model_kwargs)
        return source + velocity

    def p_sample_loop(
        self,
        model,
        shape,
        cond,
        source,
        y=None,
        num_steps=1,
        device=None,
        progress=True,
        clip_denoised=True,
        model_kwargs=None,
        solver="euler",
        discrepancy_map=None,
    ):
        del shape
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        if solver == "direct" or int(num_steps) <= 1:
            output = self.direct_prediction(model, cond, source, y, device, model_kwargs, discrepancy_map)
            if clip_denoised:
                output = self._clip_wavelets(output)
            return output

        x_t = source.clone()
        timesteps = th.linspace(0.0, 1.0, int(num_steps) + 1, device=device)
        iterator = range(int(num_steps))
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"WFM {solver.upper()} ({num_steps} steps)")

        for i in iterator:
            x_t = self.p_sample_step(
                model=model,
                x_t=x_t,
                t_curr=timesteps[i].item(),
                t_next=timesteps[i + 1].item(),
                cond=cond,
                source=source,
                y=y,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                solver=solver,
                discrepancy_map=discrepancy_map,
            )
        if clip_denoised:
            x_t = self._clip_wavelets(x_t)
        return x_t

    def _clip_wavelets(self, x):
        x_idwt = apply_idwt(x)
        return apply_dwt(x_idwt.clamp(0.0, 1.0))

    def sample_for_evaluation(
        self,
        model,
        batch,
        contr="t1c",
        num_steps=1,
        device=None,
        progress=False,
        clip_denoised=True,
        solver="euler",
    ):
        if device is None:
            device = next(model.parameters()).device

        target, conditions, class_idx = self._get_target_and_conditions(batch, contr)
        cond_dwt, source_dwt, _ = self._prepare_condition(conditions, contr)

        discrepancy_map = None
        if self.use_discrepancy:
            discrepancy_map = self.discrepancy_predictor(cond_dwt)
            if self.discrepancy_input == "target":
                # At inference time the target is unavailable.  The recovered
                # implementation uses the predictor output in this case.
                discrepancy_map = discrepancy_map

        b = target.shape[0]
        y = th.full((b,), class_idx, device=device, dtype=th.long)
        output_dwt = self.p_sample_loop(
            model=model,
            shape=source_dwt.shape,
            cond=cond_dwt,
            source=source_dwt,
            y=y,
            num_steps=num_steps,
            device=device,
            progress=progress,
            clip_denoised=clip_denoised,
            solver=solver,
            discrepancy_map=discrepancy_map,
        )
        output_img = apply_idwt(output_dwt).clamp(0.0, 1.0)
        output_img[conditions[0] == 0] = 0.0
        return output_img, target

