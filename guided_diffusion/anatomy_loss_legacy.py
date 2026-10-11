"""
Anatomical consistency loss for medical image synthesis.
Uses a simple 3D CNN encoder to extract multi-scale anatomical features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomyFeatureExtractor(nn.Module):
    """
    Simple 3D CNN encoder to extract multi-scale anatomical features.
    Outputs feature maps at 4 different scales.
    """

    def __init__(self, in_channels=1):
        super().__init__()

        # Four downsampling stages
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Freeze all parameters (they are randomly initialized, but we keep them fixed)
        for param in self.parameters():
            param.requires_grad = False

        print(f"✅ AnatomyFeatureExtractor initialized with 4 downsampling stages")

    def forward(self, x):
        """
        Extract multi-scale anatomical features.
        Args:
            x: Input image [B, 1, D, H, W]
        Returns:
            features: List of 4 feature maps at different scales
        """
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        return features


class AnatomyConsistencyLoss(nn.Module):
    """
    Enforce anatomical consistency between generated and real images.
    """

    def __init__(self, device, loss_type='mse'):
        super().__init__()
        self.extractor = AnatomyFeatureExtractor(in_channels=1).to(device)
        self.loss_type = loss_type

        # The extractor is frozen
        for param in self.extractor.parameters():
            param.requires_grad = False

    def forward(self, pred_img, target_img):
        """
        Args:
            pred_img: Generated image [B, 1, D, H, W]
            target_img: Ground truth image [B, 1, D, H, W]
        Returns:
            anat_loss: Anatomical consistency loss (scalar)
        """
        # Extract features at multiple scales
        pred_feats = self.extractor(pred_img)
        target_feats = self.extractor(target_img)

        # Compute loss at each scale
        total_loss = 0.0
        for p_feat, t_feat in zip(pred_feats, target_feats):
            if self.loss_type == 'mse':
                loss = F.mse_loss(p_feat, t_feat)
            elif self.loss_type == 'l1':
                loss = F.l1_loss(p_feat, t_feat)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            total_loss = total_loss + loss

        return total_loss / len(pred_feats)