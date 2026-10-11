"""
Wavelet Flow Matching for 3D multi-modal MRI synthesis.

Forward process: x_t = (1-t)*x_source + t*x_target + σ(t)*noise
Model predicts: velocity v = x_target - x_source

Loss (Eq. 3): L = E[||f_θ(x_t, c, t, y) - (x_target - x_source)||²]

Unified model with class conditioning for all 4 BraTS modalities.
Added DAGCN-style discrepancy map prediction.
"""

import torch
import torch as th
import torch.nn as nn
import numpy as np
from .nn import mean_flat

from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt = DWT_3D('haar')
idwt = IDWT_3D('haar')

# Modality to class index mapping for unified model
MODALITY_TO_CLASS = {'t1n': 0, 't1c': 1, 't2w': 2, 't2f': 3}
CLASS_TO_MODALITY = {0: 't1n', 1: 't1c', 2: 't2w', 3: 't2f'}
ALL_MODALITIES = ['t1n', 't1c', 't2w', 't2f']


def apply_dwt(img):
    """Apply 3D DWT and concatenate subbands along channel dimension."""
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(img)
    return th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


def apply_idwt(x):
    """Apply 3D IDWT to reconstruct image from wavelet subbands."""
    B, _, H, W, D = x.size()
    return idwt(
        x[:, 0, :, :, :].view(B, 1, H, W, D) * 3.,
        x[:, 1, :, :, :].view(B, 1, H, W, D),
        x[:, 2, :, :, :].view(B, 1, H, W, D),
        x[:, 3, :, :, :].view(B, 1, H, W, D),
        x[:, 4, :, :, :].view(B, 1, H, W, D),
        x[:, 5, :, :, :].view(B, 1, H, W, D),
        x[:, 6, :, :, :].view(B, 1, H, W, D),
        x[:, 7, :, :, :].view(B, 1, H, W, D)
    )


def build_source_prior(cond_1_dwt, cond_2_dwt, cond_3_dwt, contr,
                       mode='mean', weights=None):
    """Build the informed source from three conditioning modalities.

    For target t1c, conditioning order is (t1n, t2w, t2f). The t1n_anchor
    mode uses a T1n-dominant prior while retaining lesion contrast cues from
    T2w/T2-FLAIR. All other targets retain the official equal-mean prior.
    """
    if mode == 'mean':
        return (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0
    if mode == 't1n_anchor' and contr == 't1c':
        w = (0.6, 0.2, 0.2) if weights is None else tuple(float(v) for v in weights)
        if len(w) != 3:
            raise ValueError(f'source_prior_weights must have 3 values, got {w}')
        if abs(sum(w) - 1.0) > 1e-4:
            raise ValueError(f'source_prior_weights must sum to 1, got {w}')
        return w[0] * cond_1_dwt + w[1] * cond_2_dwt + w[2] * cond_3_dwt
    if mode != 't1n_anchor':
        raise ValueError(f'Unknown source prior mode: {mode}')
    return (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0


class WaveletFlowMatching:
    """
    Wavelet Flow Matching for paired image-to-image translation.

    Forward process: x_t = (1-t)*x_source + t*x_target + σ(t)*noise
    Model predicts:  velocity v = x_target - x_source
    """

    def __init__(
            self,
            num_timesteps=1000,
            sigma_max=0.5,
            mode='i2i',
            use_discrepancy=False,
            enh_weight=0.0,
            source_prior_mode='mean',
            source_prior_weights=None,
            et_predictor=None,
            et_weight=1.0,
            et_as_input=False,
            dual_state=False,
            et_attention=False,
    ):
        """
        Initialize Wavelet Flow Matching.

        Args:
            num_timesteps: Number of discrete timesteps (for compatibility with U-Net)
            sigma_max: Maximum noise level (peaks at t=0.5)
            mode: Operating mode, 'i2i' for image-to-image translation
            use_discrepancy: Whether to use DAGCN-style discrepancy map
        """
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max
        self.mode = mode
        self.use_discrepancy = use_discrepancy
        self.enh_weight = enh_weight
        self.source_prior_mode = source_prior_mode
        self.source_prior_weights = source_prior_weights
        self.et_predictor = et_predictor
        self.et_weight = et_weight
        self.et_as_input = et_as_input
        self.dual_state = dual_state
        self.et_attention = et_attention
        self.last_et_state = None
        self.last_et_prob = None
        self.source_prior_weights = source_prior_weights

        # Precompute sigma schedule (peaks at t=0.5)
        t = np.linspace(0, 1, num_timesteps)
        self.sigmas = sigma_max * np.sqrt(t * (1 - t))

        # ========== DAGCN: 差异图预测模块 ==========
        if self.use_discrepancy:
            # 轻量级 3D CNN 从条件模态预测差异图
            # 输入: [B, 24, D/2, H/2, W/2] (3个条件模态的小波系数)
            # 输出: [B, 1, D/2, H/2, W/2]
            self.discrepancy_predictor = nn.Sequential(
                nn.Conv3d(24, 64, kernel_size=3, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 1, kernel_size=3, padding=1),
                nn.Sigmoid()  # 输出 0~1 概率图
            )
        # ========================================

    def compute_et_prob(self, cond_dwt, contr):
        if self.et_predictor is None:
            self.last_et_prob = None
            return None
        if contr != 't1c':
            self.last_et_prob = torch.zeros(
                (cond_dwt.shape[0], 1, *cond_dwt.shape[2:]),
                device=cond_dwt.device, dtype=cond_dwt.dtype)
            self.last_et_state = torch.zeros_like(self.last_et_prob)
            return self.last_et_prob
        device = cond_dwt.device
        if next(self.et_predictor.parameters()).device != device:
            self.et_predictor = self.et_predictor.to(device)
        self.et_predictor.eval()
        with torch.no_grad():
            logits = self.et_predictor(cond_dwt, torch.zeros(cond_dwt.shape[0], device=device))
            if logits.shape[1] > 1:
                logits = logits[:, 0:1]
            self.last_et_prob = torch.sigmoid(logits).detach()
        if self.dual_state:
            self.last_et_state = (self.last_et_prob.mean(dim=(2, 3, 4), keepdim=True) > 0.3).float()
            self.last_et_state = self.last_et_state.expand(-1, 1, *self.last_et_prob.shape[2:])
        else:
            self.last_et_state = torch.zeros_like(self.last_et_prob)
        return self.last_et_prob

    def get_sigma(self, t):
        """
        Flow matching noise schedule: σ(t) = σ_max * sqrt(t * (1-t))
        Peaks at t=0.5, zero at endpoints.

        This schedule ensures:
        - At t=0: x_t = x_source (no noise, pure source)
        - At t=1: x_t = x_target (no noise, pure target)
        - At t=0.5: Maximum uncertainty/noise

        Args:
            t: timestep in [0, 1], shape [B] or scalar
        Returns:
            sigma: noise level, same shape as t
        """
        return self.sigma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def q_sample(self, x_source, x_target, t, noise=None):
        """
        Forward process: interpolate source -> target with noise.

        x_t = (1 - t) * x_source + t * x_target + σ(t) * noise

        Args:
            x_source: wavelet coefficients of source [B, 8, D, H, W]
            x_target: wavelet coefficients of target [B, 8, D, H, W]
            t: timestep in [0, 1], shape [B]
            noise: optional pre-sampled noise

        Returns:
            x_t: noisy interpolation at time t
        """
        if noise is None:
            noise = torch.randn_like(x_target)

        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t_broadcast = t.view(-1, 1, 1, 1, 1)

        # Linear interpolation between source and target
        x_t = (1 - t_broadcast) * x_source + t_broadcast * x_target

        # Add noise scaled by sigma(t)
        sigma_t = self.get_sigma(t).view(-1, 1, 1, 1, 1)
        x_t = x_t + sigma_t * noise

        return x_t

    def _scale_timesteps(self, t):
        """
        Scale continuous timesteps [0,1] to discrete range [0, num_timesteps].
        This maintains compatibility with the pretrained U-Net timestep embedding.
        """
        return t * self.num_timesteps

    def _get_target_and_conditions(self, x_start, contr):
        """
        Get target and conditioning modalities based on target contrast.

        Args:
            x_start: dict with 't1n', 't1c', 't2w', 't2f' tensors
            contr: target contrast name or class index

        Returns:
            target: target modality tensor
            conditions: list of 3 conditioning modality tensors
            class_idx: integer class index for the target modality
        """
        # Handle both string and integer inputs
        if isinstance(contr, int):
            class_idx = contr
            contr = CLASS_TO_MODALITY[contr]
        else:
            class_idx = MODALITY_TO_CLASS[contr]

        # Get target modality
        target = x_start[contr]

        # Get conditioning modalities (all others)
        conditions = [x_start[m] for m in ALL_MODALITIES if m != contr]

        return target, conditions, class_idx

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None,
                        mode='i2i', contr='t1n'):
        """
        Compute flow matching training loss (Eq. 3 from paper).

        Loss: L = E[||f_θ(x_t, c, t, y) - (x_target - x_source)||²]

        Model learns velocity field v = x_target - x_source.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Get target, conditions, and class index
        target, conditions, class_idx = self._get_target_and_conditions(x_start, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions and concatenate [B, 24, D/2, H/2, W/2]
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)
        self.compute_et_prob(cond_dwt, contr)
        if self.et_attention and self.last_et_prob is not None:
            cond_dwt = cond_dwt * (1.0 + self.last_et_prob)

        # Apply DWT to target [B, 8, D/2, H/2, W/2]
        target_dwt = apply_dwt(target)

        # === V10: 与训练一致的等权先验 ===
        source_dwt = build_source_prior(
            cond_1_dwt, cond_2_dwt, cond_3_dwt, contr,
            mode=self.source_prior_mode,
            weights=self.source_prior_weights,
        )

        # ========== DAGCN（V6.6 修正） ==========
        # 模型输入的第33通道 = 真差异图（训练期的"标准答案"）
        # predictor 输出只用于外部辅助损失，逼它模仿真差异图
        discrepancy_map = None  # predictor 输出 → 返回给训练脚本算损失
        model_disc = None  # 实际喂给模型的通道
        if self.use_discrepancy:
            if next(self.discrepancy_predictor.parameters()).device != cond_dwt.device:
                self.discrepancy_predictor = self.discrepancy_predictor.to(cond_dwt.device)
            discrepancy_map = self.discrepancy_predictor(cond_dwt)  # [B,1,D/2,H/2,W/2]
            model_disc = th.abs(target_dwt[:, 0:1] - source_dwt[:, 0:1]).detach()
        # =========================================

        # Convert discrete timestep to continuous t ∈ [0, 1]
        t_continuous = t.float() / self.num_timesteps

        # Sample noise and apply DWT
        if noise is None:
            noise = th.randn_like(target)
        noise_dwt = apply_dwt(noise)

        # Forward: interpolate source -> target with noise
        x_t = self.q_sample(source_dwt, target_dwt, t_continuous, noise=noise_dwt)

        # ========== 修改：拼接差异图作为额外条件 ==========
        if self.et_as_input and self.last_et_prob is not None:
            if self.dual_state and self.last_et_state is not None:
                x_t_cond = th.cat([x_t, cond_dwt, self.last_et_prob, self.last_et_state], dim=1)
            else:
                x_t_cond = th.cat([x_t, cond_dwt, self.last_et_prob], dim=1)
        elif self.use_discrepancy and model_disc is not None:
            # x_t [B, 8] + cond_dwt [B, 24] + 真差异图 [B, 1] = [B, 33]
            x_t_cond = th.cat([x_t, cond_dwt, model_disc], dim=1)
        else:
            x_t_cond = th.cat([x_t, cond_dwt], dim=1)
        # ==================================================

        # Create class labels for modality conditioning [B]
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=target.device, dtype=th.long)

        # Model predicts velocity v = x_target - x_source
        t_scaled = self._scale_timesteps(t_continuous)
        model_output = model(x_t_cond, t_scaled, y=y, **model_kwargs)

        # Velocity loss (Eq. 3): ||f_θ(x_t) - (x_target - x_source)||²
        velocity = target_dwt - source_dwt
        sq_err = (velocity - model_output) ** 2

        # === 强化区加权（仅 t1c 目标）：强化体素的 velocity 误差权重放大 ===
        if contr == 't1c' and self.enh_weight > 0:
            # cond_1 此时是 t1n；图像域算 mask，avg_pool 下采样到 DWT 网格
            enh = ((target - cond_1) > 0.15).float()  # [B,1,32,512,512]
            enh = th.nn.functional.avg_pool3d(enh, kernel_size=2)  # -> [B,1,16,256,256] 软权重
            w = 1.0 + self.enh_weight * enh
            mse_per_channel = mean_flat(sq_err * w) / mean_flat(w.expand_as(sq_err))
        else:
            mse_per_channel = mean_flat(sq_err)
        loss = th.mean(mse_per_channel, dim=0)
        # ==============================================================

        # For visualization: predicted target = source + velocity
        pred_target = source_dwt + model_output
        model_output_idwt = apply_idwt(pred_target)

        terms = {"mse_wav": loss}

        # === MODIFIED: 返回 source_dwt 供外部差异图监督使用 ===
        return terms, model_output, model_output_idwt, target_dwt, discrepancy_map, source_dwt
        # =====================================================

    def p_sample_step_euler(self, model, x_t, t_curr, t_next, cond, source, y=None,
                            model_kwargs=None, deterministic=True, discrepancy_map=None):
        """Euler integration step. Model predicts velocity directly."""
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        device = x_t.device

        t_model = torch.full((B,), t_curr * self.num_timesteps, device=device)

        # ========== 拼接差异图（如果存在） ==========
        if self.use_discrepancy and discrepancy_map is not None:
            x_t_cond = torch.cat([x_t, cond, discrepancy_map], dim=1)
        else:
            x_t_cond = torch.cat([x_t, cond], dim=1)
        # ========================================

        with torch.no_grad():
            velocity = model(x_t_cond, t_model, y=y, **model_kwargs)

        dt = t_next - t_curr
        x_next = x_t + dt * velocity

        return x_next

    def p_sample_step_heun(self, model, x_t, t_curr, t_next, cond, source, y=None,
                           model_kwargs=None, discrepancy_map=None):
        """Heun's method (RK2). Model predicts velocity directly."""
        if model_kwargs is None:
            model_kwargs = {}

        B = x_t.shape[0]
        device = x_t.device
        dt = t_next - t_curr

        t_model_curr = torch.full((B,), t_curr * self.num_timesteps, device=device)

        if self.use_discrepancy and discrepancy_map is not None:
            x_t_cond = torch.cat([x_t, cond, discrepancy_map], dim=1)
        else:
            x_t_cond = torch.cat([x_t, cond], dim=1)

        with torch.no_grad():
            v1 = model(x_t_cond, t_model_curr, y=y, **model_kwargs)

        x_pred = x_t + dt * v1

        t_model_next = torch.full((B,), t_next * self.num_timesteps, device=device)

        if self.use_discrepancy and discrepancy_map is not None:
            x_pred_cond = torch.cat([x_pred, cond, discrepancy_map], dim=1)
        else:
            x_pred_cond = torch.cat([x_pred, cond], dim=1)

        with torch.no_grad():
            v2 = model(x_pred_cond, t_model_next, y=y, **model_kwargs)

        x_next = x_t + dt * 0.5 * (v1 + v2)

        return x_next

    def p_sample_step(self, model, x_t, t_curr, t_next, cond, source, y=None,
                      model_kwargs=None, clip_denoised=True, solver='euler',
                      discrepancy_map=None):
        """Single sampling step using specified solver."""
        if solver == 'heun':
            return self.p_sample_step_heun(model, x_t, t_curr, t_next, cond, source, y,
                                           model_kwargs, discrepancy_map)
        else:
            return self.p_sample_step_euler(model, x_t, t_curr, t_next, cond, source, y,
                                            model_kwargs, True, discrepancy_map)

    def direct_prediction(self, model, cond, source, y=None, device=None,
                          model_kwargs=None, discrepancy_map=None):
        """Single-step prediction: source + velocity."""
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        B = source.shape[0]
        t_model = torch.zeros(B, device=device, dtype=torch.float32)

        if self.use_discrepancy and discrepancy_map is not None:
            x_cond = torch.cat([source, cond, discrepancy_map], dim=1)
        else:
            x_cond = torch.cat([source, cond], dim=1)

        with torch.no_grad():
            velocity = model(x_cond, t_model, y=y, **model_kwargs)

        return source + velocity

    def p_sample_loop(
            self,
            model,
            shape,
            cond,
            source,
            y=None,
            num_steps=50,
            device=None,
            progress=True,
            clip_denoised=True,
            model_kwargs=None,
            solver='euler',
            discrepancy_map=None,
    ):
        """Generate samples by integrating the learned velocity field."""
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device

        # Direct prediction mode - single forward pass
        if solver == 'direct' or num_steps == 1:
            output = self.direct_prediction(model, cond, source, y, device,
                                            model_kwargs, discrepancy_map)
            if clip_denoised:
                output = self._clip_wavelets(output)
            return output

        # Start from source
        x_t = source.clone()

        # Timesteps from 0 to 1
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        if progress:
            from tqdm.auto import tqdm
            solver_name = solver.upper()
            iterator = tqdm(range(num_steps), desc=f"WFM {solver_name} ({num_steps} steps)")
        else:
            iterator = range(num_steps)

        for i in iterator:
            t_curr = timesteps[i].item()
            t_next = timesteps[i + 1].item()

            x_t = self.p_sample_step(
                model=model,
                x_t=x_t,
                t_curr=t_curr,
                t_next=t_next,
                cond=cond,
                source=source,
                y=y,
                model_kwargs=model_kwargs,
                clip_denoised=clip_denoised,
                solver=solver,
                discrepancy_map=discrepancy_map,
            )

        # Optional: clip in image domain for final output
        if clip_denoised:
            x_t = self._clip_wavelets(x_t)

        return x_t

    def _clip_wavelets(self, x):
        """Clip wavelets via image domain clamping to [0,1]."""
        x_idwt = apply_idwt(x)
        x_idwt_clamp = x_idwt.clamp(0., 1.)
        return apply_dwt(x_idwt_clamp)

    def sample_for_evaluation(
            self,
            model,
            batch,
            contr='t1n',
            num_steps=50,
            device=None,
            progress=True,
            clip_denoised=True,
    ):
        """Sample a modality from batch dict, returns (synthesized, ground_truth)."""
        if device is None:
            device = next(model.parameters()).device

        # Get target, conditions, and class index using helper
        target, conditions, class_idx = self._get_target_and_conditions(batch, contr)
        cond_1, cond_2, cond_3 = conditions

        # Apply DWT to conditions
        cond_1_dwt = apply_dwt(cond_1)
        cond_2_dwt = apply_dwt(cond_2)
        cond_3_dwt = apply_dwt(cond_3)
        cond_dwt = th.cat([cond_1_dwt, cond_2_dwt, cond_3_dwt], dim=1)

        # ========== 预测差异图（用于采样） ==========
        discrepancy_map = None
        if self.use_discrepancy:
            # 确保 discrepancy_predictor 与 cond_dwt 在同一设备
            if next(self.discrepancy_predictor.parameters()).device != cond_dwt.device:
                self.discrepancy_predictor = self.discrepancy_predictor.to(cond_dwt.device)
            discrepancy_map = self.discrepancy_predictor(cond_dwt)
        # ========================================

        # === V10: 论文原版等权信息先验，所有方向统一 ===
        source_dwt = (cond_1_dwt + cond_2_dwt + cond_3_dwt) / 3.0
        # =======================================

        # Create class labels for modality conditioning
        batch_size = target.shape[0]
        y = th.full((batch_size,), class_idx, device=device, dtype=th.long)

        # Run flow matching sampling
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
            discrepancy_map=discrepancy_map,
        )

        # Convert to image domain
        output_img = apply_idwt(output_dwt)

        # Clamp and mask
        output_img = output_img.clamp(0., 1.)
        output_img[cond_1 == 0] = 0  # Zero out non-brain regions

        return output_img, target