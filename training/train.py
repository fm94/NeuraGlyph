import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from alphabet.alphabet import NUM_CLASSES
from loss.loss import ShapeCoherenceLoss
from models.decrypter import Decrypter
from models.encrypter import Encryper
from utils.io import dump_samples, create_timestamped_dir


def train_networks(config: argparse.Namespace, device: torch.device) -> tuple[Encryper, Decrypter, Path]:
    """
    Train encrypter and decrypter neural networks.

    Args:
        config (argparse.Namespace): Configuration parameters.
        device (torch.device): The device (CPU or GPU) on which the models and tensors will be loaded.

    Returns:
        tuple[Encryper, Decrypter, Path]: Trained encrypter and decrypter networks in addition to the dump dir.
    """
    dump_dir = create_timestamped_dir(config.dump_dir)

    config.batch_size = NUM_CLASSES * 2

    # Define neural networks and losses
    encryter = Encryper(noise_dim=config.noise_size, num_classes=NUM_CLASSES, temperature=config.temperature,
                        image_size=config.image_size).to(device)

    shape_loss = ShapeCoherenceLoss(connectivity_weight=config.connectivity, sparsity_weight=config.sparsity,
                                    smoothness_weight=config.smoothness, line_weight=config.line_width,
                                    target_density=config.density)

    decryter = Decrypter(num_classes=NUM_CLASSES, image_size=config.image_size).to(device)
    decryter_loss = nn.CrossEntropyLoss()

    enc_optimizer = optim.Adam(encryter.parameters(), lr=config.lr)
    dec_optimizer = optim.Adam(decryter.parameters(), lr=config.lr)

    # For visualization during training
    fixed_noise = torch.randn(NUM_CLASSES, config.noise_size, device=device)
    fixed_labels = torch.arange(NUM_CLASSES, device=device)

    enc_losses = []
    dec_losses = []

    print(">>> Starting training...")
    for epoch in range(config.num_epochs):
        noise = torch.randn(config.batch_size, config.noise_size, device=device)
        labels = torch.tensor(sum([[i] * (config.batch_size // NUM_CLASSES) for i in range(NUM_CLASSES)], []),
                              device=device)

        # Train encrypter
        new_images = encryter(noise, labels, binary=False)
        chars_loss = shape_loss(new_images)

        dec_outputs = decryter(new_images)
        dec_loss = decryter_loss(dec_outputs, labels)

        enc_total_loss = chars_loss * config.gc_tradeoff_loss + dec_loss

        enc_optimizer.zero_grad()
        enc_total_loss.backward()
        enc_optimizer.step()

        # Train decrypter with binary images
        with torch.no_grad():
            new_images = encryter(noise, labels, binary=True)

        dec_outputs = decryter(new_images)
        dec_loss = decryter_loss(dec_outputs, labels)

        dec_optimizer.zero_grad()
        dec_loss.backward()
        dec_optimizer.step()

        enc_losses.append(enc_total_loss.item())
        dec_losses.append(dec_loss.item())

        # Log losses and sample every 100 epochs
        if (epoch + 1) % 100 == 0:
            _, predicted = torch.max(dec_outputs, 1)
            accuracy = (predicted == labels).float().mean()
            print(
                f">>> Epoch [{epoch + 1}/{config.num_epochs}] | Encryter Loss: {enc_total_loss.item():.4f} | Decryter Loss: {dec_loss.item():.4f} | Decryter Accuracy: {accuracy.item():.4f}")

            with torch.no_grad():
                new_images = encryter(fixed_noise, fixed_labels, binary=True)
                dump_samples(new_images, fixed_labels, f"Samples - Epoch {epoch + 1}", dump_dir / f"epoch_{epoch}.png")

    print(">>> Training finished!")

    # Save the trained models at the end of training
    torch.save(encryter.state_dict(), dump_dir / 'encrypter_model.pth')
    torch.save(decryter.state_dict(), dump_dir / 'decrypter_model.pth')
    print(">>> Models saved to disk.")

    return encryter, decryter, dump_dir
