import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from prdc import compute_prdc
from scipy import linalg
from torch.utils.data import DataLoader, Subset
try:
    from torchmetrics.image.inception import InceptionV3
except ImportError:  # fallback for older torchmetrics releases
    from torchmetrics.image.fid import InceptionV3
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from utils import load_model


MNIST_IMAGE_SHAPE = (1, 28, 28)
FEATURE_DIM = 2048


@dataclass
class Stats:
    mean: np.ndarray
    cov: np.ndarray
    features: np.ndarray


def parse_checkpoint_specs(values: Sequence[str]) -> List[Tuple[str, Path]]:
    checkpoints: List[Tuple[str, Path]] = []
    for spec in values:
        if "=" not in spec:
            raise ValueError(f"Invalid checkpoint spec '{spec}'. Use name=/path/to/checkpoints.")
        label, path_str = spec.split("=", 1)
        label = label.strip()
        checkpoint_path = Path(path_str).expanduser().resolve()
        if not label:
            raise ValueError(f"Checkpoint label missing in '{spec}'.")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory '{checkpoint_path}' not found for '{label}'.")
        if not (checkpoint_path / "G.pth").exists():
            raise FileNotFoundError(f"G.pth missing in '{checkpoint_path}' for '{label}'.")
        checkpoints.append((label, checkpoint_path))
    return checkpoints


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_inception(device: torch.device) -> InceptionV3:
    model = InceptionV3(output_blocks=[3], normalize_input=False)
    model.to(device)
    model.eval()
    return model


def preprocess_for_inception(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.dtype != torch.float32:
        images = images.float()
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return images.to(device)


def compute_dataset_subset(
    dataset: datasets.MNIST, limit: int, batch_size: int, device: torch.device
) -> Iterable[torch.Tensor]:
    actual_limit = min(limit, len(dataset))
    subset = Subset(dataset, range(actual_limit))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    for images, _ in loader:
        yield images


def collect_real_features(
    device: torch.device,
    inception: InceptionV3,
    num_samples: int,
    batch_size: int,
    data_root: Path,
    split: str,
) -> Stats:
    dataset = datasets.MNIST(
        root=str(data_root),
        train=(split == "train"),
        transform=transforms.ToTensor(),
        download=True,
    )
    features: List[np.ndarray] = []

    iterator = compute_dataset_subset(dataset, num_samples, batch_size, device)
    for batch in tqdm(iterator, desc=f"real-{split}", unit="batch"):
        images = batch.to(device)
        prepared = preprocess_for_inception(images, device)
        with torch.inference_mode():
            activations = inception(prepared)[0].view(images.size(0), -1)
        features.append(activations.cpu().numpy())

    stacked = np.concatenate(features, axis=0)
    return Stats(mean=np.mean(stacked, axis=0), cov=np.cov(stacked, rowvar=False), features=stacked)


def save_sample_grid(samples: torch.Tensor, destination: Path, max_samples: int = 64) -> None:
    if max_samples <= 0:
        return
    total = samples.size(0)
    if total == 0:
        return
    grid_count = min(total, max_samples)
    grid = samples[:grid_count]
    nrow = int(math.sqrt(grid_count))
    nrow = max(1, nrow)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, destination, nrow=nrow)


def collect_fake_features(
    label: str,
    checkpoint_dir: Path,
    device: torch.device,
    inception: InceptionV3,
    num_samples: int,
    batch_size: int,
    latent_dim: int,
    store_samples: bool,
    output_dir: Path,
) -> Stats:
    generator = Generator(g_output_dim=np.prod(MNIST_IMAGE_SHAPE))
    generator = load_model(generator, str(checkpoint_dir), device)
    generator.to(device)
    generator.eval()

    features: List[np.ndarray] = []
    saved_batches: List[torch.Tensor] = []
    total = 0

    progress = tqdm(total=num_samples, desc=f"gen-{label}", unit="img")
    while total < num_samples:
        current = min(batch_size, num_samples - total)
        z = torch.randn(current, latent_dim, device=device)
        with torch.inference_mode():
            generated = generator(z).view(current, *MNIST_IMAGE_SHAPE)
        images = ((generated + 1.0) / 2.0).clamp(0.0, 1.0)
        if store_samples:
            saved_count = sum(batch.size(0) for batch in saved_batches)
            if saved_count < store_samples:
                remaining = store_samples - saved_count
                saved_batches.append(images[:remaining].cpu())

        prepared = preprocess_for_inception(images, device)
        with torch.inference_mode():
            activations = inception(prepared)[0].view(current, -1)
        features.append(activations.cpu().numpy())

        total += current
        progress.update(current)
    progress.close()

    if store_samples:
        stacked = torch.cat(saved_batches, dim=0) if saved_batches else torch.empty(0)
        if stacked.numel() > 0:
            grid_path = output_dir / label / "samples_grid.png"
            save_sample_grid(stacked, grid_path, max_samples=store_samples)

    stacked = np.concatenate(features, axis=0)
    return Stats(mean=np.mean(stacked, axis=0), cov=np.cov(stacked, rowvar=False), features=stacked)


def compute_frechet_distance(real_stats: Stats, fake_stats: Stats) -> float:
    diff = real_stats.mean - fake_stats.mean
    cov_prod = real_stats.cov.dot(fake_stats.cov)
    covmean, _ = linalg.sqrtm(cov_prod, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(real_stats.cov.shape[0]) * 1e-6
        covmean = linalg.sqrtm((real_stats.cov + offset).dot(fake_stats.cov + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(real_stats.cov + fake_stats.cov - 2 * covmean)
    return float(fid)


def compute_precision_recall(real_stats: Stats, fake_stats: Stats, nearest_k: int) -> Dict[str, float]:
    effective_k = min(nearest_k, real_stats.features.shape[0] - 1, fake_stats.features.shape[0] - 1)
    if effective_k < 1:
        raise ValueError("Not enough samples to compute precision/recall. Increase num_samples.")
    prdc_scores = compute_prdc(
        real_features=real_stats.features,
        fake_features=fake_stats.features,
        nearest_k=effective_k,
    )
    return {
        "precision": float(prdc_scores["precision"]),
        "recall": float(prdc_scores["recall"]),
        "density": float(prdc_scores["density"]),
        "coverage": float(prdc_scores["coverage"]),
        "nearest_k": float(effective_k),
    }


def evaluate_checkpoints(
    checkpoints: Sequence[Tuple[str, Path]],
    args: argparse.Namespace,
    device: torch.device,
    real_stats: Stats,
    inception: InceptionV3,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, ckpt in checkpoints:
        fake_stats = collect_fake_features(
            label=label,
            checkpoint_dir=ckpt,
            device=device,
            inception=inception,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            store_samples=args.sample_grid,
            output_dir=output_dir,
        )
        fid = compute_frechet_distance(real_stats, fake_stats)
        pr_metrics = compute_precision_recall(real_stats, fake_stats, args.pr_nearest_k)

        combined = {"fid": fid}
        combined.update(pr_metrics)
        combined["num_samples"] = float(fake_stats.features.shape[0])
        combined["checkpoint_dir"] = str(ckpt)
        metrics[label] = combined

        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(combined, fp, indent=2)

    return metrics


def default_checkpoints() -> List[Tuple[str, Path]]:
    defaults: List[Tuple[str, Path]] = []
    baseline = Path("checkpoints")
    diffaug = Path("checkpoints_diffaug")

    if (baseline / "G.pth").exists():
        defaults.append(("baseline", baseline))
    if (diffaug / "G.pth").exists():
        defaults.append(("diffaug", diffaug))
    return defaults


def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    header = f"{'model':<12} {'FID':>10} {'precision':>12} {'recall':>10} {'density':>10} {'coverage':>10}"
    lines = [header, "-" * len(header)]
    for label, metrics in results.items():
        lines.append(
            f"{label:<12} "
            f"{metrics['fid']:>10.4f} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['density']:>10.4f} "
            f"{metrics['coverage']:>10.4f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FID/precision/recall metrics for MNIST GAN checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        metavar="LABEL=PATH",
        help="Named checkpoint directory containing G.pth (can be provided multiple times).",
    )
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples per model.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for feature extraction.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension for the generator.")
    parser.add_argument(
        "--real-split",
        choices=("train", "test"),
        default="test",
        help="MNIST split to use as the real distribution reference.",
    )
    parser.add_argument(
        "--pr-nearest-k",
        type=int,
        default=5,
        help="k for improved precision and recall metric (PRDC).",
    )
    parser.add_argument(
        "--sample-grid",
        type=int,
        default=64,
        help="Number of generated samples to store in a grid image (0 to disable).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/docker_eval",
        help="Directory where metrics and sample grids are written.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for downloading MNIST.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generator sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoints:
        checkpoints = parse_checkpoint_specs(args.checkpoints)
    else:
        checkpoints = default_checkpoints()
        if not checkpoints:
            raise ValueError(
                "No checkpoints provided and defaults missing. "
                "Use --checkpoint label=path to specify at least one model."
            )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    print(f"Using device: {device}")

    inception = build_inception(device)
    real_stats = collect_real_features(
        device=device,
        inception=inception,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        data_root=Path(args.data_root),
        split=args.real_split,
    )
    results = evaluate_checkpoints(checkpoints, args, device, real_stats, inception)
    print(format_metrics_table(results))


if __name__ == "__main__":
    main()
