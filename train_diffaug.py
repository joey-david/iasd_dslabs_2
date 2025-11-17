import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from model import Generator, Discriminator
from utils import save_models


N_CRITIC = 2
BETA1 = 0.5
MNIST_SHAPE = (1, 28, 28)
DEFAULT_POLICY = "blur,bleed,localized_noise"
DEFAULT_CHECKPOINT_DIR = "checkpoints_diffaug"


@dataclass
class DiffAugmentParams:
    blur_strength: float = 3.0
    blur_kernel_max: int = 9
    bleed_strength: float = 1.0
    bleed_length: float = 7.0
    bleed_decay: float = 1.7
    bleed_angle_deg: float = 15.0
    noise_strength: float = 2.5


_AUGMENT_PARAMS = DiffAugmentParams()
_AUGMENT_FNS = {}


def _register_diffaugment_ops() -> None:
    """Register differentiable augmentations tailored to MNIST strokes."""

    def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _shift_tensor(x: torch.Tensor, dx: int, dy: int, fill: float = 0.0) -> torch.Tensor:
        shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        if dy > 0:
            shifted[:, :, :dy, :] = fill
        elif dy < 0:
            shifted[:, :, dy:, :] = fill
        if dx > 0:
            shifted[:, :, :, :dx] = fill
        elif dx < 0:
            shifted[:, :, :, dx:] = fill
        return shifted

    def rand_blur(x: torch.Tensor) -> torch.Tensor:
        params = _AUGMENT_PARAMS
        if x.size(2) < 3 or x.size(3) < 3 or params.blur_strength <= 0:
            return x
        max_dim = min(max(3, params.blur_kernel_max), x.size(2), x.size(3))
        if max_dim < 3:
            return x
        kernel_size = torch.randint(3, max_dim + 1, (1,), device=x.device).item()
        if kernel_size % 2 == 0:
            kernel_size = max(3, kernel_size - 1)
        sigma = (0.7 + torch.rand(1, device=x.device).item()) * (1.0 + 0.3 * params.blur_strength)
        kernel = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.size(1), 1, 1, 1)
        pad = kernel_size // 2
        padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        blurred = F.conv2d(padded, kernel, padding=0, groups=x.size(1))
        base_mix = torch.empty(1, device=x.device).uniform_(0.5, 0.95).item()
        strength_scale = min(params.blur_strength, 4.0)
        mix = base_mix * (0.8 + 0.2 * strength_scale)
        mix = float(max(0.0, min(mix, 0.98)))
        return (1 - mix) * x + mix * blurred

    def bleed_whites(x: torch.Tensor) -> torch.Tensor:
        params = _AUGMENT_PARAMS
        if params.bleed_strength <= 0:
            return x

        whiteness = (x + 1.0) * 0.5
        accum = whiteness.clone()
        weight_sum = torch.ones_like(whiteness)
        angle = math.radians(params.bleed_angle_deg)
        direction = (math.cos(angle), math.sin(angle))
        max_steps = max(1, int(round(params.bleed_length)))
        decay = max(params.bleed_decay, 1e-3)

        for step in range(1, max_steps + 1):
            dx = int(round(direction[0] * step))
            dy = int(round(direction[1] * step))
            if dx == 0 and dy == 0:
                continue
            weight = math.exp(-(step - 1) / decay)
            shifted = _shift_tensor(whiteness, dx, dy)
            accum = accum + weight * shifted
            weight_sum = weight_sum + weight

        bleed = accum / weight_sum
        scaled = whiteness + params.bleed_strength * (bleed - whiteness)
        scaled = torch.clamp(scaled, 0.0, 1.0)
        return scaled * 2.0 - 1.0

    def localized_noise(x: torch.Tensor) -> torch.Tensor:
        params = _AUGMENT_PARAMS
        if params.noise_strength <= 0:
            return x
        whiteness = (x + 1.0) * 0.5
        gauss_kernel = _gaussian_kernel(5, 1.0, x.device, x.dtype)
        gauss_kernel = gauss_kernel.view(1, 1, 5, 5).repeat(x.size(1), 1, 1, 1)
        closeness = F.conv2d(whiteness, gauss_kernel, padding=2, groups=x.size(1))
        stroke_presence = torch.clamp(whiteness * 1.5, 0.0, 1.0)
        region_mask = F.max_pool2d(stroke_presence, kernel_size=11, stride=1, padding=5)
        distance_weight = torch.clamp(closeness * region_mask, 0.0, 1.0)
        noise = torch.rand_like(x) * 2.0 - 1.0
        augmented = x + params.noise_strength * distance_weight * noise
        return augmented.clamp(-1, 1)

    global _AUGMENT_FNS
    _AUGMENT_FNS = {
        "blur": [rand_blur],
        "bleed": [bleed_whites],
        "localized_noise": [localized_noise],
    }


_register_diffaugment_ops()


def apply_diffaugment(x: torch.Tensor, policy: str = DEFAULT_POLICY) -> torch.Tensor:
    if not policy:
        return x
    output = x
    for p in policy.split(","):
        p = p.strip()
        if not p:
            continue
        for fn in _AUGMENT_FNS.get(p, []):
            output = fn(output)
    return output.clamp(-1, 1)


def list_diffaugment_ops() -> Tuple[str, ...]:
    return tuple(_AUGMENT_FNS.keys())


def discriminator_forward(images: torch.Tensor, D: nn.Module, policy: str) -> torch.Tensor:
    augmented = apply_diffaugment(images, policy)
    flattened = augmented.view(images.size(0), -1)
    return D(flattened)


def calculate_gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
    policy: str = DEFAULT_POLICY,
) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator_forward(interpolates, D, policy)
    grad_outputs = torch.ones_like(disc_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * lambda_gp


def d_step(
    real_images: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    policy: str,
) -> Tuple[float, float]:
    D.zero_grad()
    batch_size = real_images.size(0)
    real_logits = discriminator_forward(real_images, D, policy)
    d_loss_real = -real_logits.mean()

    z = torch.randn(batch_size, 100, device=device)
    fake_vectors = G(z)
    fake_images = fake_vectors.view(batch_size, *MNIST_SHAPE)

    fake_logits = discriminator_forward(fake_images.detach(), D, policy)
    d_loss_fake = fake_logits.mean()

    gradient_penalty = calculate_gradient_penalty(D, real_images, fake_images.detach(), device, policy=policy)

    d_loss = d_loss_real + d_loss_fake + gradient_penalty
    d_loss.backward()
    optimizer.step()

    wasserstein_distance = real_logits.mean() - fake_logits.mean()
    return d_loss.item(), wasserstein_distance.item()


def g_step(
    batch_size: int,
    G: nn.Module,
    D: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    policy: str,
) -> float:
    G.zero_grad()
    z = torch.randn(batch_size, 100, device=device)
    fake_vectors = G(z)
    fake_images = fake_vectors.view(batch_size, *MNIST_SHAPE)
    fake_logits = discriminator_forward(fake_images, D, policy)
    g_loss = -fake_logits.mean()
    g_loss.backward()
    optimizer.step()
    return g_loss.item()


def create_dataloaders(batch_size: int, data_root: str, download: bool) -> Iterable[torch.Tensor]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=download)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader


def prepare_device(requested_gpus: int) -> Tuple[torch.device, str, int]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if requested_gpus == -1:
            requested_gpus = torch.cuda.device_count()
        return device, "cuda", requested_gpus
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps", 1
    return torch.device("cpu"), "cpu", 1


def train_diffaug(
    epochs: int,
    lr: float,
    batch_size: int,
    checkpoint_dir: Path,
    policy: str,
    requested_gpus: int,
    max_steps: Optional[int] = None,
) -> None:
    device, device_label, effective_gpus = prepare_device(requested_gpus)
    print(f"Using device: {device_label}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    data_root = os.getenv("DATA", "data")
    download = not Path(data_root, "MNIST").exists()
    loader = create_dataloaders(batch_size, data_root, download)

    mnist_dim = 28 * 28
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    if device_label == "cuda" and effective_gpus > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(BETA1, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(BETA1, 0.999))

    global_step = 0

    epoch_iterator = tqdm(range(1, epochs + 1), desc="diffaug-epochs", leave=False)
    for epoch in epoch_iterator:
        batch_iterator = tqdm(loader, desc=f"epoch-{epoch}", leave=False)
        for real_images, _ in batch_iterator:
            real_images = real_images.to(device)

            d_loss, w_dist = d_step(real_images, G, D, d_optimizer, device, policy)
            batch_iterator.set_postfix(d_loss=d_loss, w_dist=w_dist)

            if global_step % N_CRITIC == 0:
                g_loss = g_step(real_images.size(0), G, D, g_optimizer, device, policy)
                batch_iterator.set_postfix(d_loss=d_loss, g_loss=g_loss, w_dist=w_dist)

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        if max_steps is not None and global_step >= max_steps:
            break

        if epoch % 10 == 0:
            save_models(G, D, str(checkpoint_dir))

    save_models(G, D, str(checkpoint_dir))
    print(f"Training complete. Checkpoints stored in {checkpoint_dir}")


def ensure_diffaug_weights(checkpoint_dir: Path, requested_gpus: int = -1) -> None:
    g_ckpt = checkpoint_dir / "G.pth"
    d_ckpt = checkpoint_dir / "D.pth"
    if g_ckpt.exists() and d_ckpt.exists():
        print(f"DiffAugment checkpoints found in {checkpoint_dir}.")
        return

    auto_env = os.getenv("DIFFAUG_AUTOTRAIN", "1").lower()
    if auto_env not in {"1", "true", "yes"}:
        print(
            f"DiffAugment checkpoints missing in {checkpoint_dir}, "
            "but auto-training is disabled via DIFFAUG_AUTOTRAIN."
        )
        return

    print("DiffAugment checkpoints missing. Launching training run.")
    train_diffaug(
        epochs=150,
        lr=1e-4,
        batch_size=64,
        checkpoint_dir=checkpoint_dir,
        policy=DEFAULT_POLICY,
        requested_gpus=requested_gpus,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST GAN with DiffAugment (Zhao et al., 2020).")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory.")
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY, help="DiffAugment policy string.")
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available). Ignored on CPU/MPS.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional cap on total training steps for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_diffaug(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
        policy=args.policy,
        requested_gpus=args.gpus,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
