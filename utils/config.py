import argparse


def read_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepEncrypt V1.0")
    parser.add_argument('--dump_dir', type=str, default="data", help="Where to store weights, training logs and images")
    parser.add_argument('--image_size', type=int, default=16, help="Image size per letter (must be a multiple of 8!)")
    parser.add_argument('--num_epochs', type=int, default=1000, help="Number of epochs used for training")
    parser.add_argument('--noise_size', type=int, default=100, help="Input noise dimension")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate, applicable for all nets")
    parser.add_argument('--gc_tradeoff_loss', type=float, default=0.5,
                        help="Tradeoff between decrypter and encrypter loss. Used to train the encrypter")
    parser.add_argument('--connectivity', type=float, default=15., help="Character connectivity weight")
    parser.add_argument('--sparsity', type=float, default=1., help="Character sparsity weight")
    parser.add_argument('--smoothness', type=float, default=0.1, help="Character smoothness weight")
    parser.add_argument('--line_width', type=float, default=0.00002, help="Character line width")
    parser.add_argument('--density', type=float, default=0.6, help="Character character space density")
    parser.add_argument('--temperature', type=float, default=0.5,
                        help="Temperature factor to scale output layer the sigmoid and have more extreme values")
    parser.add_argument('--train', type=bool, nargs='?', const=True, default=False, help="Training mode")
    parser.add_argument('--encrypt', type=bool, nargs='?', const=True, default=False, help="Encryption mode")
    parser.add_argument('--decrypt', type=bool, nargs='?', const=True, default=False, help="Decryption mode")
    parser.add_argument('--text_file', type=str, default="data/example_text.txt",
                        help="Text file where the text to encrypt is")
    parser.add_argument('--image_file', type=str, default="data/example_image.png", help="Image to decrypt")
    args = parser.parse_args()
    config_check(args)
    print(">>> Config loaded.")
    return args


def config_check(config: argparse.Namespace) -> None:
    """
    Check whether the inference parameters are consistent.

    Args:
        config (argparse.Namespace): Configuration parameters.
    """
    if not (config.train or config.encrypt or config.decrypt):
        raise ValueError("At least one task must be chosen: [train, encrypt, decrypt]")  # add others checks...
