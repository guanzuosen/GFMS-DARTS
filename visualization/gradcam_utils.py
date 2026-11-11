"""Dependency-light, standard Grad-CAM utilities.

The implementation deliberately uses PyTorch hooks instead of a third-party
Grad-CAM package.  This makes the recorded activation/gradient pair explicit
for the project-specific NAS Cells and DIDS modules.
"""

from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _resample_bilinear():
    """Return a Pillow bilinear enum on both old and new Pillow versions."""
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


class GradCAM(object):
    """Compute Grad-CAM for one module whose output is a 4-D feature map."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.dim() != 4:
            raise RuntimeError(
                "Grad-CAM target layer must output a 4-D [B,C,H,W] tensor; "
                "received {!r}.".format(type(output))
            )
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        gradient = grad_output[0]
        if gradient is None or gradient.dim() != 4:
            raise RuntimeError("Grad-CAM target layer did not receive a 4-D output gradient.")
        self.gradients = gradient

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __call__(self, inputs, target_class, output_size):
        """Return a normalized [H,W] NumPy CAM for a single input image.

        ``target_class`` must be the ground-truth integer class supplied by the
        caller.  The function intentionally never substitutes model predictions.
        """
        if inputs.size(0) != 1:
            raise ValueError("GradCAM currently expects exactly one input sample.")
        if not isinstance(target_class, int):
            raise TypeError("target_class must be an integer ground-truth class.")
        self.activations = None
        self.gradients = None
        self.model.zero_grad()
        logits = self.model(inputs)
        if logits.dim() != 2 or not 0 <= target_class < logits.size(1):
            raise ValueError(
                "Invalid Grad-CAM target class {} for logits with shape {}.".format(
                    target_class, tuple(logits.shape)
                )
            )
        logits[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        if tuple(self.activations.shape) != tuple(self.gradients.shape):
            raise RuntimeError(
                "Grad-CAM activation/gradient shapes differ: {} versus {}.".format(
                    tuple(self.activations.shape), tuple(self.gradients.shape)
                )
            )

        # Standard Grad-CAM: spatially average gradients, weight feature maps,
        # sum channels and retain positive class evidence.
        channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(output_size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().float().cpu().numpy()
        return normalize_cam(cam)


def normalize_cam(cam, eps=1e-12):
    """Normalize one raw CAM to [0,1] without changing its spatial support."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or not np.isfinite(cam).all():
        raise RuntimeError("Grad-CAM contains invalid (non-finite or non-2D) values.")
    minimum = float(cam.min())
    maximum = float(cam.max())
    if maximum - minimum <= eps:
        raise RuntimeError("Grad-CAM is spatially constant; refusing to save an all-black/all-red map.")
    normalized = (cam - minimum) / (maximum - minimum)
    if not np.isfinite(normalized).all():
        raise RuntimeError("Grad-CAM normalization produced non-finite values.")
    return normalized.astype(np.float32, copy=False)


def tensor_to_display_image(tensor, mean, std):
    """Invert Normalize on a [3,H,W] tensor and return an RGB uint8 image."""
    if tensor.dim() != 3 or tensor.size(0) != 3:
        raise ValueError("Expected an RGB [3,H,W] tensor for display.")
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = (tensor.detach() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def jet_colormap(cam):
    """A fixed jet-like colour map without relying on matplotlib."""
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    # The standard piecewise "jet" construction, expressed vectorially.
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0.0, 1.0)
    return np.stack((r, g, b), axis=-1)


def overlay_cam(rgb_image, cam, alpha=0.45):
    """Overlay a normalized CAM on an RGB uint8 image using the fixed jet map."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1].")
    image = np.asarray(rgb_image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must be uint8 RGB.")
    if tuple(cam.shape) != tuple(image.shape[:2]):
        cam = np.asarray(
            Image.fromarray(np.asarray(cam * 255.0, dtype=np.uint8)).resize(
                (image.shape[1], image.shape[0]), _resample_bilinear()
            ),
            dtype=np.float32,
        ) / 255.0
    heatmap = jet_colormap(cam)
    blended = (1.0 - alpha) * (image.astype(np.float32) / 255.0) + alpha * heatmap
    return np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8)


def resize_rgb(rgb_image, size):
    return np.asarray(Image.fromarray(rgb_image).resize(tuple(size), _resample_bilinear()), dtype=np.uint8)

