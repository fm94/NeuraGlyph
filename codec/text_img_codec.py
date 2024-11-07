import numpy as np
import torch
from PIL import Image

from alphabet.alphabet import IDX_TO_CHAR, CHAR_TO_IDX
from models.decrypter import Decrypter
from models.encrypter import Encryper


class TextVisualCodec:
    """
    A class to encrypt and decrypt text to and from images.
    
    Attributes:
        encrypter (Encryper): The encrypter model used to generate images from text.
        decrypter (Decrypter): The decrypter model used to decode images back into text.
        device (torch.device): The device (CPU or GPU) on which the models and tensors will be loaded.
        noise_dim (int): The noise dimension used by the encrypter model.
        image_size (int): The size of the image generated by the encrypter.
    """

    def __init__(self, encrypter: Encryper, decrypter: Decrypter, device: torch.device, image_size: int):
        """
        Initialize the TextVisualCodec with an encrypter, decrypter, device, and image size.

        Args:
            encrypter (Encryper): The encrypter model.
            decrypter (Decrypter): The decrypter model.
            device (torch.device): The device to use for tensor operations.
            image_size (int): The size of the image generated by the encrypter.
        """
        self.encrypter = encrypter
        self.decrypter = decrypter
        self.device = device
        self.noise_dim = encrypter.noise_dim
        self.image_size = image_size

    def encrypt_text(self, text_file: str, image_file: str) -> None:
        """
        Convert text from a file to a grid of images representing the characters.

        Args:
            text_file (str): The path to the text file to encode.
            image_file: path to the image file to be saved.
        
        Raises:
            ValueError: If the text contains no valid characters to encode.
        """
        with open(text_file, 'r') as file:
            text = file.read()

        text = text.upper()
        # Convert characters to indices
        char_indices = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]

        if not char_indices:
            raise ValueError("No valid characters to encode")

        # Generate images for each character
        noise = torch.randn(len(char_indices), self.noise_dim, device=self.device)
        labels = torch.tensor(char_indices, device=self.device)

        with torch.no_grad():
            images = self.encrypter(noise, labels, binary=True)

        # Arrange images in grid (max 10 per row)
        images = images.cpu().numpy()
        n_chars = len(char_indices)
        sqrt_n_chars = int(np.sqrt(n_chars)) + 1
        rows = (n_chars + sqrt_n_chars - 1) // sqrt_n_chars
        cols = min(n_chars, sqrt_n_chars)

        # Create final image grid
        grid = np.ones((rows * self.image_size, cols * self.image_size))
        for idx, img in enumerate(images):
            row = idx // sqrt_n_chars
            col = idx % sqrt_n_chars
            grid[row * self.image_size:(row + 1) * self.image_size,
            col * self.image_size:(col + 1) * self.image_size] = img.squeeze()

        # Dump into disk
        grid_image = Image.fromarray((grid * 255).astype(np.uint8))
        grid_image.save(image_file, format='PNG')
        print(f">>> Text encrypted and dumped into {image_file}")

    def decrypt_image(self, image_file: str) -> None:
        """
        Convert an image grid back into text by decoding each character image.

        Args:
            image_file (str): The path to the image file to decode.
        """
        image = Image.open(image_file)

        # Convert to a NumPy array
        grid = np.array(image) / 255

        # Split grid into individual character images
        rows = grid.shape[0] // self.image_size
        cols = grid.shape[1] // self.image_size

        chars = []
        for row in range(rows):
            for col in range(cols):
                # Extract character image
                img = grid[row * self.image_size:(row + 1) * self.image_size,
                      col * self.image_size:(col + 1) * self.image_size]
                if not np.all(img == 1):  # Skip empty cells
                    # Convert to tensor and classify
                    img_tensor = torch.tensor(img, device=self.device).float().unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        output = self.decrypter(img_tensor)
                        predicted = output.argmax().item()
                        chars.append(IDX_TO_CHAR[predicted])

        print(f">>> Decrypted text: {''.join(chars)}")
