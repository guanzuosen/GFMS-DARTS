import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "OPS", "MSTF_NAS_SPACE_NORMAL", "MSTF_NAS_SPACE_DARTS", "MSTF_NAS_SPACE_PDC",
    "MSTF_NAS_SPACE_REDUCT", "PixelDifferenceConv2d", "PDCSepConv",
    "PDCDepthwiseSeparableConv", "APDCDepthwiseSeparableConv", "DIDS",
]

# 使用字典构建Lambda函数，调用相应Operations
OPS = {
    # Original DARTS Normal Cell operations.
    'none'             : lambda c, affine : Zero(),
    'max_pool_3x3'     : lambda c, affine : nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    'skip_connect'     : lambda c, affine : nn.Identity(),
    'avg_pool_3x3'     : lambda c, affine : nn.AvgPool2d(
        kernel_size=3, stride=1, padding=1, count_include_pad=False
    ),
    'sep_conv_3x3'     : lambda c, affine : SepConv(
        in_c=c, out_c=c, kernel_s=3, stride=1, padding=1, affine=affine
    ),
    'sep_conv_5x5'     : lambda c, affine : SepConv(
        in_c=c, out_c=c, kernel_s=5, stride=1, padding=2, affine=affine
    ),
    'dil_conv_3x3'     : lambda c, affine : DilSepConv(
        in_c=c, out_c=c, kernel_s=3, stride=1, padding=2, dilation=2, affine=affine
    ),
    'dil_conv_5x5'     : lambda c, affine : DilSepConv(
        in_c=c, out_c=c, kernel_s=5, stride=1, padding=4, dilation=2, affine=affine
    ),
    # Single-stage PDC operations for the Normal Cell search space.
    'pdc_sep_conv_3x3': lambda c, affine : PDCDepthwiseSeparableConv(
        in_c=c, out_c=c, kernel_s=3, stride=1, padding=1, affine=affine, theta=0.5
    ),
    'pdc_sep_conv_5x5': lambda c, affine : PDCDepthwiseSeparableConv(
        in_c=c, out_c=c, kernel_s=5, stride=1, padding=2, affine=affine, theta=0.5
    ),
    'conv_1x1': lambda c, affine : ReLUConvBN(
        in_c=c, out_c=c, kernel_s=1, stride=1, padding=0, affine=affine
    ),
    'asym_conv_7x7': lambda c, affine : ReLUDualConvBN(
        in_c=c, out_c=c, kernel_s=7, stride=1, padding=3, affine=affine
    ),
    # Original project Reduction Cell operations.
    'avg_com_pool_2x2' : lambda in_c, out_c, affine : CompressPooling(2, 2, 0, in_c, out_c, 'Avg', affine),
    'max_com_pool_2x2' : lambda in_c, out_c, affine : CompressPooling(2, 2, 0, in_c, out_c, 'Max', affine),
    'avg_com_pool_3x3' : lambda in_c, out_c, affine : CompressPooling(3, 2, 1, in_c, out_c, 'Avg', affine),
    'max_com_pool_3x3' : lambda in_c, out_c, affine : CompressPooling(3, 2, 1, in_c, out_c, 'Max', affine),
    'avg_com_pool_4x4' : lambda in_c, out_c, affine : CompressPooling(4, 2, 1, in_c, out_c, 'Avg', affine),
    'max_com_pool_4x4' : lambda in_c, out_c, affine : CompressPooling(4, 2, 1, in_c, out_c, 'Max', affine),
}

MSTF_NAS_SPACE_DARTS = [
    'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
    'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
]

MSTF_NAS_SPACE_PDC = [
    # PDC3 + PDC5 + Conv1 + Asmy7 + Skip + None + Dil3 + Dil5.
    'pdc_sep_conv_3x3',
    'pdc_sep_conv_5x5',
    'conv_1x1',
    'asym_conv_7x7',
    'skip_connect',
    'none',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

# Backward-compatible alias for the complete proposed method (E4).
MSTF_NAS_SPACE_NORMAL = MSTF_NAS_SPACE_PDC

MSTF_NAS_SPACE_REDUCT = [
    'avg_com_pool_2x2', 'max_com_pool_2x2', 'avg_com_pool_3x3',
    'max_com_pool_3x3', 'avg_com_pool_4x4', 'max_com_pool_4x4',
]


'''Will be used in the Normal Cell'''
# ReLU + Conv + BN
class ReLUConvBN(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=kernel_s, 
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# ReLU + (7x1) + (1x7) + BN when ``kernel_s=7``.
class ReLUDualConvBN(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=in_c, kernel_size=(kernel_s, 1),
                stride=(stride, 1), padding=(padding, 0), bias=False
            ),
            nn.Conv2d(
                in_channels=in_c, out_channels=out_c, kernel_size=(1, kernel_s),
                stride=(1, stride), padding=(0, padding), bias=False
            ),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# Dilation Separable Convolution
class DilSepConv(nn.Module):
    def __init__(self,
        in_c, out_c, kernel_s, stride, padding, dilation=1, affine=True
    ):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=in_c, out_channels=in_c, dilation=dilation, groups=in_c,       # 当groups等于通道数是，即可分离卷积
                kernel_size=kernel_s, stride=stride, padding=padding, bias=False
            ),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_c, affine=affine),
        )
    
    def forward(self, Inputs):
        return self.main(Inputs)

# Separable Convolution
class SepConv(nn.Module):
    def __init__(self, 
        in_c, out_c, kernel_s, stride, padding, affine=True
    ):
        super(SepConv, self).__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=kernel_s, stride=stride, padding=padding, groups=in_c, bias=False),
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_c, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_c, in_c, kernel_size=kernel_s, stride=1, padding=padding, groups=in_c, bias=False),
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )

    def forward(self, Inputs):
        return self.main(Inputs)


class PixelDifferenceConv2d(nn.Module):
    """Depth-wise central pixel-difference convolution.

    For every neighbour x_i, this layer learns weights for (x_i - x_center),
    making the response explicitly sensitive to local intensity differences.
    """
    def __init__(self, channels, kernel_s, stride=1, padding=0, theta=0.7):
        super().__init__()
        self.theta = theta
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_s, stride=stride,
            padding=padding, groups=channels, bias=False
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        if self.theta == 0:
            return output

        # sum(W) * x_center is the central term of sum(W * (x_i - x_center)).
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        center = F.conv2d(
            inputs, kernel_diff, stride=self.conv.stride, padding=0,
            groups=self.conv.groups
        )
        return output - self.theta * center


class PDCSepConv(nn.Module):
    """Two-stage depth-wise separable convolution using pixel differences."""
    def __init__(self, in_c, out_c, kernel_s, stride, padding, affine=True, theta=0.7):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            PixelDifferenceConv2d(in_c, kernel_s, stride=stride, padding=padding, theta=theta),
            nn.Conv2d(in_c, in_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_c, affine=affine),
            nn.ReLU(inplace=False),
            PixelDifferenceConv2d(in_c, kernel_s, stride=1, padding=padding, theta=theta),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )

    def forward(self, inputs):
        return self.main(inputs)


class PDCDepthwiseSeparableConv(nn.Module):
    """Single-stage PDC depth-wise separable convolution.

    This is intentionally lighter than ``PDCSepConv``: it performs one PDC
    depth-wise convolution followed by one 1x1 point-wise projection.
    """
    def __init__(self, in_c, out_c, kernel_s, stride, padding, affine=True, theta=0.7):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            PixelDifferenceConv2d(in_c, kernel_s, stride=stride, padding=padding, theta=theta),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )

    def forward(self, inputs):
        return self.main(inputs)


class AngularDifferenceConv2d(nn.Module):
    """Original APDC: clockwise angular pixel differences in a 3x3 neighbourhood."""
    def __init__(self, channels, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, stride=stride, padding=padding,
            dilation=dilation, groups=channels, bias=False
        )

    def forward(self, inputs):
        # The permutation follows the official PiDiNet APDC implementation.
        weight = self.conv.weight
        shape = weight.shape
        weight = weight.view(shape[0], shape[1], -1)
        clockwise = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        angular_weight = (weight - weight[:, :, clockwise]).view(shape)
        return F.conv2d(
            inputs, angular_weight, stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups
        )


class APDCDepthwiseSeparableConv(nn.Module):
    """Single-stage APDC depth-wise separable convolution from PiDiNet."""
    def __init__(self, in_c, out_c, stride=1, padding=1, affine=True):
        super().__init__()
        self.main = nn.Sequential(
            nn.ReLU(inplace=False),
            AngularDifferenceConv2d(in_c, stride=stride, padding=padding),
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )

    def forward(self, inputs):
        return self.main(inputs)


class DIDS(nn.Module):
    """Fixed hierarchical Avg/Max multi-scale dynamic downsampler.

    AvgPool and MaxPool each contain aligned 2x2, 3x3, and 4x4 stride-2
    branches.  Each scale gate receives the concatenated spatial mean and
    standard deviation of its three scale features, then per-sample scalar
    gates fuse scales within each pooling type and fuse the Avg/Max results.
    A single 1x1 projection maps channels after fusion.  DIDS has no
    architecture parameters and is never part of the Normal Cell
    candidate-operation pool.
    """
    def __init__(self, in_c, out_c=None, reduction=4, affine=True):
        super().__init__()
        if in_c <= 0:
            raise ValueError("DIDS in_c must be positive.")
        if reduction <= 0:
            raise ValueError("DIDS reduction must be positive.")
        out_c = 2 * in_c if out_c is None else out_c
        if out_c <= 0:
            raise ValueError("DIDS out_c must be positive.")

        self.in_c = in_c
        self.out_c = out_c
        self.reduction = reduction
        # ceil_mode keeps 2x2 and 4x4 aligned with 3x3 at the final
        # odd-sized 7x7 -> 4x4 transition; it does not alter even-size stages.
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.avg_pool_3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool_4 = nn.AvgPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, ceil_mode=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(in_c // reduction, 4)
        self.avg_scale_gate = nn.Sequential(
            nn.Linear(6 * in_c, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.max_scale_gate = nn.Sequential(
            nn.Linear(6 * in_c, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.type_gate = nn.Sequential(
            nn.Linear(2 * in_c, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_c, affine=affine),
        )
        self.last_alpha_mean = None
        self.last_beta_mean = None
        self.last_gamma_mean = None
        self.last_gate_means = None
        # Per-sample gate recording is deliberately opt-in.  It is used only
        # by post-hoc analysis and is not a parameter or a search variable.
        self._record_gate_weights = False
        self.last_gate_weights = None

    def set_gate_recording(self, enabled=True):
        """Enable/disable read-only storage of the latest batch gate weights.

        When disabled (the default), ``forward`` has exactly the original
        tensor return type and does not retain per-sample gate tensors.  When
        enabled, the stored tensors are detached from autograd and can be read
        after a normal ``forward`` call via :meth:`get_last_gate_weights`.
        """
        self._record_gate_weights = bool(enabled)
        if not self._record_gate_weights:
            self.last_gate_weights = None

    def get_last_gate_weights(self):
        """Return the latest recorded batch gates, or ``None`` when disabled."""
        return self.last_gate_weights

    @staticmethod
    def _validate_branch_shapes(**features):
        shapes = {name: tuple(feature.shape) for name, feature in features.items()}
        if len(set(shapes.values())) != 1:
            raise RuntimeError(
                "All DIDS pooling branch shapes must match before fusion; "
                f"received {shapes}."
            )

    @staticmethod
    def _mean_std_descriptor(feature):
        """Return the per-channel spatial mean and standard deviation."""
        mean = feature.mean(dim=(2, 3))
        std = feature.std(dim=(2, 3), unbiased=False)
        return torch.cat([mean, std], dim=1)

    def _fuse_scales(self, feature_2, feature_3, feature_4, gate):
        descriptors = torch.cat(
            [
                self._mean_std_descriptor(feature_2),
                self._mean_std_descriptor(feature_3),
                self._mean_std_descriptor(feature_4),
            ],
            dim=1,
        )
        weights = torch.softmax(gate(descriptors), dim=1)
        fused = (
            weights[:, 0].view(-1, 1, 1, 1) * feature_2
            + weights[:, 1].view(-1, 1, 1, 1) * feature_3
            + weights[:, 2].view(-1, 1, 1, 1) * feature_4
        )
        return fused, weights

    def forward(self, inputs, return_gate_weights=False):
        if inputs.dim() != 4:
            raise ValueError(f"DIDS expects a 4D tensor, received shape {tuple(inputs.shape)}.")
        if inputs.shape[1] != self.in_c:
            raise ValueError(f"DIDS expected {self.in_c} input channels, received {inputs.shape[1]}.")

        avg_2 = self.avg_pool_2(inputs)
        avg_3 = self.avg_pool_3(inputs)
        avg_4 = self.avg_pool_4(inputs)
        max_2 = self.max_pool_2(inputs)
        max_3 = self.max_pool_3(inputs)
        max_4 = self.max_pool_4(inputs)
        self._validate_branch_shapes(
            avg_2=avg_2, avg_3=avg_3, avg_4=avg_4,
            max_2=max_2, max_3=max_3, max_4=max_4,
        )

        avg_fused, alpha = self._fuse_scales(avg_2, avg_3, avg_4, self.avg_scale_gate)
        max_fused, beta = self._fuse_scales(max_2, max_3, max_4, self.max_scale_gate)
        type_descriptors = torch.cat(
            [self.gap(avg_fused).flatten(1), self.gap(max_fused).flatten(1)], dim=1
        )
        gamma = torch.softmax(self.type_gate(type_descriptors), dim=1)
        fused = (
            gamma[:, 0].view(-1, 1, 1, 1) * avg_fused
            + gamma[:, 1].view(-1, 1, 1, 1) * max_fused
        )

        self.last_alpha_mean = alpha.detach().mean(dim=0)
        self.last_beta_mean = beta.detach().mean(dim=0)
        self.last_gamma_mean = gamma.detach().mean(dim=0)
        self.last_gate_means = {
            "alpha": self.last_alpha_mean,
            "beta": self.last_beta_mean,
            "gamma": self.last_gamma_mean,
        }
        if self._record_gate_weights:
            self.last_gate_weights = {
                "avg_scale": alpha.detach(),
                "max_scale": beta.detach(),
                "type": gamma.detach(),
            }

        out = self.proj(fused)
        if return_gate_weights:
            return out, {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "avg_scale": alpha,
                "max_scale": beta,
                "type": gamma,
            }
        return out

class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Inputs):
        return Inputs.mul(0.0)


'''Will be used in the Reduct Cell'''
class CompressPooling(nn.Module):
    def __init__(self,
        kernel_size, stride, padding, in_c, out_c, 
        pool_type:str='Max',affine=True,
    ):
        super().__init__()
        if pool_type not in ['Avg', 'Max']:
            ValueError("The type of pooling isn't valid! It only can be 'Avg' or 'Max'!")
        self.pool     = nn.ModuleList()
        self.pool.append(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding) if pool_type == 'Max'
            else nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False),
        )
        if kernel_size % 2:     # 当kernel_size为奇数时，增添一个向下取整的pool操作
            self.pool.append(
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding-1) if pool_type == 'Max'
                else nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding-1, count_include_pad=False),
            )
        self.compress = nn.Identity() if in_c == out_c else ReLUConvBN(in_c=in_c, out_c=out_c, kernel_s=1, stride=1, padding=0, affine=affine)
    
    def forward(self, Inputs):
        # 当height和width为奇数时，选择向下取整的pool操作
        Inputs = self.pool[-1](Inputs) if (Inputs.shape[-1] % 2) else self.pool[0](Inputs)
        return self.compress(Inputs)
