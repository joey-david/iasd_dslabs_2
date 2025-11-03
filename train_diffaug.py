import argparse
import os
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
DEFAULT_POLICY = "translation,cutout"
DEFAULT_CHECKPOINT_DIR = "checkpoints_diffaug"

_AUGMENT_FNS = {}


def _register_diffaugment_ops() -> None:
    """Register differentiable augmentations from Zhao et al., NeurIPS 2020."""

    def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
        if ratio <= 0:
            return x

        batch, channels, height, width = x.shape
        max_shift_h = int(height * ratio + 0.5)
        max_shift_w = int(width * ratio + 0.5)
        if max_shift_h == 0 and max_shift_w == 0:
            return x

        # Reflective padding lets us slice translated crops without grid_sample,
        # which avoids backends missing the grid_sampler backward implementation.
        pad = (max_shift_w, max_shift_w, max_shift_h, max_shift_h)
        padded = F.pad(x, pad, mode="reflect")

        shift_h = torch.randint(
            -max_shift_h, max_shift_h + 1, (batch,), device=x.device
        )
        shift_w = torch.randint(
            -max_shift_w, max_shift_w + 1, (batch,), device=x.device
        )

        translated = []
        for idx in range(batch):
            h_start = max_shift_h + shift_h[idx].item()
            w_start = max_shift_w + shift_w[idx].item()
            translated.append(
                padded[idx : idx + 1, :, h_start : h_start + height, w_start : w_start + width]
            )

        return torch.cat(translated, dim=0)

    def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
        if ratio <= 0:
            return x
        height = x.size(2)
        width = x.size(3)
        cutout_h = max(1, int(height * ratio + 0.5))
        cutout_w = max(1, int(width * ratio + 0.5))

        offset_x = torch.randint(0, height, (x.size(0), 1, 1), device=x.device)
        offset_y = torch.randint(0, width, (x.size(0), 1, 1), device=x.device)
        grid_x = torch.arange(height, device=x.device).view(1, -1, 1)
        grid_y = torch.arange(width, device=x.device).view(1, 1, -1)
        mask_x = (grid_x - offset_x).abs() > cutout_h // 2
        mask_y = (grid_y - offset_y).abs() > cutout_w // 2
        mask = (mask_x | mask_y).to(x.dtype).unsqueeze(1)
        return x * mask

    global _AUGMENT_FNS
    _AUGMENT_FNS = {
        "translation": [rand_translation],
        "cutout": [rand_cutout],
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


def discriminator_forward(images: torch.Tensor, D: nn.Module, policy: str) -> torch.Tensor:
    augmented = apply_diffaugment(images, policy)
    return D(augmented)


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
    fake_images = G(z)

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
    fake_images = G(z)
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

    G = Generator().to(device)
    D = Discriminator().to(device)

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
