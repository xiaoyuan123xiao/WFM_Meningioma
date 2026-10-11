"""Multi-scale anatomical consistency loss for WFM ablation."""

import torch.nn as nn
import torch.nn.functional as F


class AnatomyFeatureExtractor(nn.Module):
    """Randomly initialized, frozen multi-scale 3D feature extractor.

    ``arch='paper'`` follows the manuscript description:
    Conv3d(3x3x3) -> InstanceNorm3d -> ReLU -> MaxPool3d(2x2x2).

    ``arch='legacy'`` follows the recovered experimental file:
    Conv3d(3x3x3, stride=2) -> ReLU.
    """

    def __init__(self, in_channels=1, arch="paper"):
        super().__init__()
        if arch not in {"paper", "legacy"}:
            raise ValueError("arch must be 'paper' or 'legacy'")
        self.arch = arch

        channels = (16, 32, 64, 128)
        self.features = nn.ModuleList()
        prev = in_channels
        for out_channels in channels:
            if arch == "paper":
                block = nn.Sequential(
                    nn.Conv3d(prev, out_channels, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool3d(kernel_size=2),
                )
            else:
                block = nn.Sequential(
                    nn.Conv3d(prev, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            self.features.append(block)
            prev = out_channels

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        outputs = []
        for block in self.features:
            x = block(x)
            outputs.append(x)
        return outputs


class AnatomyConsistencyLoss(nn.Module):
    """Enforce feature-space consistency.  The feature extractor is frozen."""

    def __init__(self, device, loss_type="mse", arch="paper"):
        super().__init__()
        self.extractor = AnatomyFeatureExtractor(in_channels=1, arch=arch).to(device)
        self.loss_type = loss_type
        for parameter in self.extractor.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        # Keep the random feature extractor in eval mode so InstanceNorm does
        # not update running statistics while the generator is training.
        super().train(mode)
        self.extractor.eval()
        return self

    def forward(self, pred_img, target_img):
        pred_features = self.extractor(pred_img)
        target_features = self.extractor(target_img)

        total = pred_img.new_zeros(())
        for pred_feature, target_feature in zip(pred_features, target_features):
            if self.loss_type == "mse":
                total = total + F.mse_loss(pred_feature, target_feature)
            elif self.loss_type == "l1":
                total = total + F.l1_loss(pred_feature, target_feature)
            else:
                raise ValueError(f"Unknown anatomy loss type: {self.loss_type}")
        return total / len(pred_features)
