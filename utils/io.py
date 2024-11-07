import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from alphabet.alphabet import IDX_TO_CHAR, NUM_CLASSES
from models.decrypter import Decrypter
from models.encrypter import Encryper


def dump_samples(images: torch.Tensor, labels: torch.Tensor, title: str, save_path: Path) -> None:
    """
    Save a grid of images with labels.

    Args:
        images (torch.Tensor): Generated images to display.
        labels (torch.Tensor): Corresponding labels for images.
        title (str): Title for the plot.
        save_path (Path): Path to save the output image.
    """
    rows = (NUM_CLASSES + 9) // 10
    fig, axes = plt.subplots(rows, 10, figsize=(10, 1 * rows))
    axes = axes.flatten()

    for idx, (img, label) in enumerate(zip(images, labels)):
        if idx < len(axes):
            axes[idx].imshow(img.cpu().squeeze(), cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(IDX_TO_CHAR[label.item()])

    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    plt.savefig(save_path, format='png', bbox_inches='tight')
    plt.close(fig)


def create_timestamped_dir(base_path: str = '.') -> Path:
    """
    Create a directory with a timestamp to save outputs.

    Args:
        base_path (str): Base path where the directory is created.

    Returns:
        Path: The created directory path.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = Path(base_path) / timestamp
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f">>> Dump directory created at: {dir_path}")
    return dir_path


def load_models(config: argparse.ArgumentParser) -> tuple[Encryper, Decrypter]:
    """
    Loads trained Encrypter and Decrypter models from disk.

    Args:
        config (argparse.ArgumentParser): Configuration object containing model parameters.

    Returns:
        tuple[Encryper, Decrypter]: The loaded Encrypter and Decrypter models.
    """
    # Initialize the models with the same config used during training
    encryter = Encryper(noise_dim=config.noise_size, num_classes=NUM_CLASSES, temperature=config.temperature,
                        image_size=config.image_size)
    decryter = Decrypter(num_classes=NUM_CLASSES, image_size=config.image_size)

    # Load the state dictionaries
    encryter.load_state_dict(torch.load(Path(config.dump_dir) / 'encrypter_model.pth', weights_only=True))
    decryter.load_state_dict(torch.load(Path(config.dump_dir) / 'decrypter_model.pth', weights_only=True))

    print(">>> Models loaded from disk.")
    return encryter, decryter
