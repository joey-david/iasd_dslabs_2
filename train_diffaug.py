import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
DEFAULT_POLICY = "translation,gaussian_blur,gaussian_noise"
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

    def _gaussian_kernel(
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords = coords - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def rand_gaussian_blur(
        x: torch.Tensor,
        kernel_size: int = 5,
        sigma_range: Tuple[float, float] = (0.4, 1.2),
        p: float = 0.7,
    ) -> torch.Tensor:
        if kernel_size % 2 == 0 or p <= 0:
            return x

        apply_mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < p
        if not apply_mask.any():
            return x

        sigma = torch.empty(1, device=x.device).uniform_(*sigma_range).item()
        kernel = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
        kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)
        blurred = F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.size(1))
        return torch.where(apply_mask, blurred, x)

    def rand_gaussian_noise(
        x: torch.Tensor,
        std_range: Tuple[float, float] = (0.05, 0.15),
        p: float = 0.8,
    ) -> torch.Tensor:
        if p <= 0:
            return x

        apply_mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < p
        if not apply_mask.any():
            return x

        noise_std = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(*std_range)
        noise = torch.randn_like(x) * noise_std
        return torch.where(apply_mask, x + noise, x)

    global _AUGMENT_FNS
    _AUGMENT_FNS = {
        "translation": [rand_translation],
        "gaussian_blur": [rand_gaussian_blur],
        "gaussian_noise": [rand_gaussian_noise],
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
    create_graph: bool = True,
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
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean() * lambda_gp


def critic_loss(
    real_images: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    device: torch.device,
    policy: str,
    *,
    create_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = real_images.size(0)
    real_logits = discriminator_forward(real_images, D, policy)
    d_loss_real = -real_logits.mean()

    z = torch.randn(batch_size, 100, device=device)
    fake_images = G(z).detach()
    fake_logits = discriminator_forward(fake_images, D, policy)
    d_loss_fake = fake_logits.mean()

    gradient_penalty = calculate_gradient_penalty(
        D, real_images, fake_images, device, policy=policy, create_graph=create_graph
    )

    d_loss = d_loss_real + d_loss_fake + gradient_penalty
    wasserstein_distance = real_logits.mean() - fake_logits.mean()
    return d_loss, wasserstein_distance


def generator_loss(
    batch_size: int,
    G: nn.Module,
    D: nn.Module,
    device: torch.device,
    policy: str,
) -> torch.Tensor:
    z = torch.randn(batch_size, 100, device=device)
    fake_images = G(z)
    fake_logits = discriminator_forward(fake_images, D, policy)
    return -fake_logits.mean()


def average(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def evaluate_losses(
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    G: nn.Module,
    D: nn.Module,
    device: torch.device,
    policy: str,
) -> Tuple[float, float]:
    was_training_g = G.training
    was_training_d = D.training
    G.eval()
    D.eval()

    critic_losses: List[float] = []
    generator_losses: List[float] = []

    for real_images, _ in loader:
        real_images = real_images.to(device)
        d_loss, _ = critic_loss(real_images, G, D, device, policy, create_graph=False)
        critic_losses.append(float(d_loss.detach().cpu().item()))

        with torch.no_grad():
            g_loss = generator_loss(real_images.size(0), G, D, device, policy)
        generator_losses.append(float(g_loss.cpu().item()))

    if was_training_g:
        G.train()
    if was_training_d:
        D.train()

    G.zero_grad(set_to_none=True)
    D.zero_grad(set_to_none=True)
    return average(critic_losses), average(generator_losses)


def save_loss_history(history: Dict[str, List[float]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)


def d_step(
    real_images: torch.Tensor,
    G: nn.Module,
    D: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    policy: str,
) -> Tuple[float, float]:
    D.zero_grad(set_to_none=True)
    d_loss, wasserstein_distance = critic_loss(real_images, G, D, device, policy)
    d_loss.backward()
    optimizer.step()

    return float(d_loss.detach().cpu().item()), float(wasserstein_distance.detach().cpu().item())


def g_step(
    batch_size: int,
    G: nn.Module,
    D: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    policy: str,
) -> float:
    G.zero_grad(set_to_none=True)
    g_loss = generator_loss(batch_size, G, D, device, policy)
    g_loss.backward()
    optimizer.step()
    return float(g_loss.detach().cpu().item())


def create_dataloaders(
    batch_size: int,
    data_root: str,
    download: bool,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=download)
    val_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader


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
    train_loader, val_loader = create_dataloaders(batch_size, data_root, download)

    G = Generator().to(device)
    D = Discriminator().to(device)

    if device_label == "cuda" and effective_gpus > 1:
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(BETA1, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(BETA1, 0.999))

    global_step = 0

    history: Dict[str, List[float]] = {
        "epochs": [],
        "train_d_loss": [],
        "train_g_loss": [],
        "train_total_loss": [],
        "train_w_distance": [],
        "val_d_loss": [],
        "val_g_loss": [],
        "val_total_loss": [],
    }

    epoch_iterator = tqdm(range(1, epochs + 1), desc="diffaug-epochs", leave=False)
    for epoch in epoch_iterator:
        train_d_losses: List[float] = []
        train_g_losses: List[float] = []
        train_wd: List[float] = []

        batch_iterator = tqdm(train_loader, desc=f"diffaug-train-{epoch}", leave=False)
        for real_images, _ in batch_iterator:
            real_images = real_images.to(device)

            d_loss, w_dist = d_step(real_images, G, D, d_optimizer, device, policy)
            train_d_losses.append(d_loss)
            train_wd.append(w_dist)

            postfix = {"d_loss": d_loss, "w_dist": w_dist}

            if global_step % N_CRITIC == 0:
                g_loss = g_step(real_images.size(0), G, D, g_optimizer, device, policy)
                train_g_losses.append(g_loss)
                postfix["g_loss"] = g_loss

            batch_iterator.set_postfix(postfix)

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        if max_steps is not None and global_step >= max_steps:
            break

        avg_train_d = average(train_d_losses)
        avg_train_g = average(train_g_losses)
        avg_train_total = avg_train_d + avg_train_g
        avg_train_wd = average(train_wd)

        val_d, val_g = evaluate_losses(val_loader, G, D, device, policy)
        val_total = val_d + val_g

        history["epochs"].append(epoch)
        history["train_d_loss"].append(avg_train_d)
        history["train_g_loss"].append(avg_train_g)
        history["train_total_loss"].append(avg_train_total)
        history["train_w_distance"].append(avg_train_wd)
        history["val_d_loss"].append(val_d)
        history["val_g_loss"].append(val_g)
        history["val_total_loss"].append(val_total)

        epoch_iterator.set_postfix(
            train_total=f"{avg_train_total:.4f}", val_total=f"{val_total:.4f}", w_dist=f"{avg_train_wd:.4f}"
        )

        if epoch % 10 == 0:
            save_models(G, D, str(checkpoint_dir))

    save_models(G, D, str(checkpoint_dir))
    save_loss_history(history, Path("results/training_logs/diffaug.json"))
    print(
        f"Training complete. Checkpoints stored in {checkpoint_dir} "
        "and loss history saved to results/training_logs/diffaug.json"
    )


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
