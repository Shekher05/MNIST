# data_utils.py
import numpy as np
import struct
from logger_utils import get_logger

logger = get_logger(__name__)

def read_images_labels(images_filepath, labels_filepath):
    """
    Reads MNIST images and labels from binary files.
    Returns:
        images: np.ndarray, shape (num_samples, 784)
        labels: np.ndarray, shape (num_samples,)
    """
    logger.info(f"Reading labels from {labels_filepath}")
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError(f"Labels file magic number mismatch: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)

    logger.info(f"Reading images from {images_filepath}")
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError(f"Images file magic number mismatch: {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows * cols)

    logger.info(f"Loaded {images.shape[0]} images of size {rows}x{cols}")
    return images, labels

def load_data_mnist_format(
    train_images_path="train-images-idx3-ubyte",
    train_labels_path="train-labels-idx1-ubyte",
    test_images_path="t10k-images-idx3-ubyte",
    test_labels_path="t10k-labels-idx1-ubyte",
    num_train=None,
    num_test=None
):
    """
    Loads MNIST data from binary files, optionally limiting the number of samples.
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    X_train, y_train = read_images_labels(train_images_path, train_labels_path)
    X_test, y_test = read_images_labels(test_images_path, test_labels_path)
    if num_train:
        X_train, y_train = X_train[:num_train], y_train[:num_train]
    if num_test:
        X_test, y_test = X_test[:num_test], y_test[:num_test]
    logger.info(f"Final train shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Final test shape: {X_test.shape}, {y_test.shape}")
    return (X_train, y_train), (X_test, y_test)