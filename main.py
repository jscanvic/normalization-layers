import copy
import random
import argparse
import os
from pathlib import Path
from contextvars import ContextVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dsets
from torch.nn import init
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# ConvNeXt LayerNorm_C definition
# -----------------------------------------------------------------------------
# Adapted from https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py#L119
class LayerNorm_C(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)


# -----------------------------------------------------------------------------
# Alias-Free LayerNorm definition
# -----------------------------------------------------------------------------
# Adapted from https://github.com/hmichaeli/alias_free_convnets/blob/9018d9858b2db44cac329c7844cbd0d873519952/models/layer_norm.py#L50
class LayerNorm_AF(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_first",
        u_dims=(1,),
        s_dims=(1, 2, 3),
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.u_dims = u_dims
        self.s_dims = s_dims

    def forward(self, x):
        if self.data_format == "channels_last":
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        if self.data_format == "channels_last":
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        init.zeros_(self.bias)


# -----------------------------------------------------------------------------
# Context variable for batch indices
# -----------------------------------------------------------------------------
current_indices: ContextVar[torch.Tensor] = ContextVar("current_indices")


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def translate(
    x: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, kind: str
) -> torch.Tensor:
    """Translate tensor spatially by specified shifts dx and dy.

    Parameters:
      x: feature map tensor of shape (N, C, H, W)
      dx: horizontal displacements, shape (N,) (must be integer dtype)
      dy: vertical displacements, shape (N,) (must be integer dtype)
      kind: either "shift" or "translate"
    """
    if kind == "shift":
        # integer pixel shifts: circular (wrap) padding
        # ensure dx and dy are integer types
        if dx.dtype not in (torch.int32, torch.int64) or dy.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise AssertionError(
                f"translate: dx and dy must be integer tensors, got {dx.dtype} and {dy.dtype}"
            )
        rolled = []
        for i in range(x.size(0)):
            # vertical shift dy -> dim -2; horizontal shift dx -> dim -1
            rolled.append(
                torch.roll(
                    x[i], shifts=(int(dy[i].item()), int(dx[i].item())), dims=(-2, -1)
                )
            )
        return torch.stack(rolled, dim=0)
    elif kind == "translate":
        N, C, H, W = x.shape
        # Forward FFT (real‑to‑complex) over spatial dims.
        X_freq = torch.fft.rfft2(x, dim=(-2, -1))  # (N, C, H, W_r)

        # Build frequency grids (cycles / pixel).
        h_freqs = torch.fft.fftfreq(H, device=x.device).view(1, 1, H, 1)  # (1,1,H,1)
        w_freqs = torch.fft.rfftfreq(W, device=x.device).view(
            1, 1, 1, -1
        )  # (1,1,1,W_r)

        # Reshape displacements for broadcasting.
        dy = dy.to(x.device).view(N, 1, 1, 1)  # (N,1,1,1)
        dx = dx.to(x.device).view(N, 1, 1, 1)  # (N,1,1,1)

        # Phase shift: exp(-2πi (k/H * dy + l/W * dx)).
        phase = torch.exp(
            -2j * torch.pi * (h_freqs * dy + w_freqs * dx)
        )  # (N,1,H,W_r) via broadcast

        # Apply phase shift to each channel (broadcast on C dimension).
        X_freq_shifted = X_freq * phase

        # Inverse FFT to recover spatial domain.
        return torch.fft.irfft2(X_freq_shifted, s=(H, W), dim=(-2, -1))
    else:
        raise ValueError(f"translate: unknown kind={kind!r}")


def compute_error(module, x, kind: str):
    """
    module: the normalization layer under test
    x:      input feature maps, shape (N, C, H, W)
    kind:   either "shift" or "translate"

    Returns:
      err: tensor of shape (N,), the equivariance error per sample
      dx:  tensor of shape (N,), the sampled horizontal displacements
      dy:  tensor of shape (N,), the sampled vertical displacements
    """
    # 1. unpack spatial dims
    N, C, H, W = x.shape

    # 2. sample a batch of random displacements
    device = x.device
    if kind == "shift":
        # use integer dx, dy directly
        dx = torch.randint(0, W, (N,), device=device, dtype=torch.int64)
        dy = torch.randint(0, H, (N,), device=device, dtype=torch.int64)
    elif kind == "translate":
        dx = torch.rand(N, device=device, dtype=x.dtype) * W
        dy = torch.rand(N, device=device, dtype=x.dtype) * H
    else:
        raise ValueError(f"compute_error: unknown kind={kind!r}")

    # 3. apply module before/after translation
    y_transformed_first = module(translate(x, dx, dy, kind))
    y_transformed_last = translate(module(x), dx, dy, kind)

    # 4. compute cosine-based error across channels, then avg spatially
    cos = F.cosine_similarity(y_transformed_first, y_transformed_last, dim=1)
    err = 1.0 - cos
    err = err.mean(dim=(-2, -1))

    return err, dx, dy


# -----------------------------------------------------------------------------
# Data loader helper
# -----------------------------------------------------------------------------


def make_imagenet_val_loader(root: Path, batch_size: int, transform):
    if transform is None:
        raise ValueError("Transform must be provided for data preprocessing.")

    class _IndexedImageNet(dsets.ImageNet):
        def __getitem__(self, index):  # type: ignore[override]
            img, target = super().__getitem__(index)
            return index, img, target

    val_set = _IndexedImageNet(str(root), split="val", transform=transform)
    return torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(os.cpu_count() or 1, 8),
        pin_memory=True,
    )


# -----------------------------------------------------------------------------
# Hook machinery
# -----------------------------------------------------------------------------


def register_equivariance_hooks(backbone: nn.Module, results: list[dict]):
    for name, m in backbone.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(make_hook(results, name))


def make_hook(results: list[dict], module_name: str):
    def hook(m: nn.BatchNorm2d, input, output):
        idxs: torch.Tensor = current_indices.get()
        x = input[0].detach()
        batch_size, _, input_height, input_width = x.shape

        # fresh normalization layers
        num_channels = m.num_features

        # BatchNorm2d copy
        new_bn = nn.BatchNorm2d(num_channels)

        # LayerNorm_CHW
        new_ln = nn.LayerNorm([num_channels, input_height, input_width])

        # InstanceNorm2d
        new_in = nn.InstanceNorm2d(num_channels)

        # LayerNorm_C
        new_lnc = LayerNorm_C(num_channels)

        # LayerNorm_AF
        new_lnaf = LayerNorm_AF(num_channels)

        norms = [new_bn, new_ln, new_in, new_lnc, new_lnaf]

        for norm in norms:
            norm = norm.to(x.device)
            layer_kind = norm.__class__.__name__
            if hasattr(norm, "weight") and norm.weight is not None:
                layer_affine = True
            elif hasattr(norm, "bias") and norm.bias is not None:
                layer_affine = True
            else:
                layer_affine = False
            if layer_affine:
                init_kinds = ["default", "normal"]
            else:
                init_kinds = ["default"]
            for init_kind in init_kinds:
                if init_kind == "default":
                    norm.reset_parameters()
                elif init_kind == "normal":
                    if hasattr(norm, "weight") and norm.weight is not None:
                        init.normal_(norm.weight, 0, 1)
                    if hasattr(norm, "bias") and norm.bias is not None:
                        init.normal_(norm.bias, 0, 1)
                else:
                    raise ValueError(f"Unknown init_kind={init_kind!r}")
                if isinstance(norm, nn.BatchNorm2d):
                    training_modes = [True, False]
                else:
                    training_modes = [False]
                for training_mode in training_modes:
                    norm.train(training_mode)
                    # Compute equivariance errors
                    for kind in ["shift", "translate"]:
                        with torch.no_grad():
                            # Save the original state of the normalization layer to make sure using it does not affect
                            # its state.
                            _norm_original = copy.deepcopy(norm)
                            err_t, dx_t, dy_t = compute_error(norm, x, kind=kind)
                            # Restore the original norm.
                            norm = _norm_original
                        for j in range(err_t.size(0)):
                            results.append(
                                {
                                    "layer_name": module_name,
                                    "image_index": int(idxs[j].item()),
                                    "input_height": input_height,
                                    "input_width": input_width,
                                    "layer_kind": layer_kind,
                                    "layer_training_mode": norm.training,
                                    "layer_affine": layer_affine,
                                    "layer_init_kind": init_kind,
                                    "transformation": kind,
                                    "horizontal_displacement": float(dx_t[j].item()),
                                    "vertical_displacement": float(dy_t[j].item()),
                                    "error": float(err_t[j].item()),
                                }
                            )

    return hook


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Translation-equivariance experiment for candidate normalisation layers."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Path to ImageNet root directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV file",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights).eval().to(device)

    samples: list[dict] = []
    register_equivariance_hooks(backbone, samples)

    transform = weights.transforms()
    loader = make_imagenet_val_loader(args.data_root, args.batch_size, transform)

    with torch.no_grad():
        for indices, imgs, _ in tqdm(loader, desc="Collecting feature maps"):
            token = current_indices.set(indices)
            backbone(imgs.to(device))
            current_indices.reset(token)

    out_path = Path(os.path.realpath(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(samples).to_csv(out_path, index=False)
    print(f"Saved per-sample equivariance errors to {out_path}")


if __name__ == "__main__":
    main()
