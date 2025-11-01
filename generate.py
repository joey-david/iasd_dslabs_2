import argparse
from pathlib import Path

import torch
import torchvision

from model import Generator, Discriminator
from utils import load_model, load_discriminator


REFINE_STEPS = 3
REFINE_STEP_SIZE = 0.05
REFINE_CLIP = 1.0


def select_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using device: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple Metal)")
        return torch.device("mps")
    print("Using device: CPU")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MNIST samples refined with discriminator gradient flow."
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Generation batch size.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to output.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory with G.pth and D.pth."
    )
    parser.add_argument(
        "--output_dir", type=str, default="samples", help="Directory to write refined samples."
    )
    return parser.parse_args()


def refine_samples(imgs: torch.Tensor, discriminator: Discriminator) -> torch.Tensor:
    refined = imgs
    discriminator.eval()
    eps = 1e-4

    for _ in range(REFINE_STEPS):
        refined = refined.detach().requires_grad_(True)
        logits = discriminator(refined.view(refined.size(0), -1))
        loss = -torch.log(logits + eps).mean()
        grad = torch.autograd.grad(loss, refined)[0]
        refined = refined - REFINE_STEP_SIZE * grad
        refined = refined.clamp(-REFINE_CLIP, REFINE_CLIP)

    return refined.detach()


def main():
    args = parse_args()
    device = select_device()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mnist_dim = 28 * 28
    generator = Generator(g_output_dim=mnist_dim).to(device)
    generator = load_model(generator, checkpoint_dir, device)
    generator.eval()

    discriminator = Discriminator(mnist_dim).to(device)
    discriminator = load_discriminator(discriminator, checkpoint_dir, device)
    discriminator.eval()

    total_generated = 0
    while total_generated < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - total_generated)
        z = torch.randn(current_batch, 100, device=device)
        with torch.no_grad():
            samples = generator(z).view(current_batch, 1, 28, 28)

        samples = refine_samples(samples, discriminator)

        for i in range(current_batch):
            filename = f"{total_generated + i}.png"
            torchvision.utils.save_image(samples[i], output_dir / filename)

        total_generated += current_batch

    print(f"Saved {args.num_samples} discriminator-refined samples to {output_dir}")


if __name__ == "__main__":
    main()
