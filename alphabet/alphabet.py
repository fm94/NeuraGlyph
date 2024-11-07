# Mapping between characters and class indices
# If you need new characters add them here
CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ?,'
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
