import torch

from codec.text_img_codec import TextVisualCodec
from training.train import train_networks
from utils.config import read_config
from utils.io import load_models

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_config()

    # training
    if config.train:
        print("### Training mode ###")
        encrypter, decrypter, config.dump_dir = train_networks(config, device)
    else:
        print("### Loading existing weights mode ###")
        encrypter, decrypter = load_models(config)

    # encryption -decryption
    codec = TextVisualCodec(encrypter=encrypter, decrypter=decrypter, device=device, image_size=config.image_size)
    if config.encrypt:
        print(">>> Encrypting...")
        codec.encrypt_text(config.text_file, config.image_file)
    if config.decrypt:
        print(">>> Decrypting...")
        codec.decrypt_image(config.image_file)
