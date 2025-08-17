import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.logger import logging
from src.exception_h import CustomException
import sys

from src.data import load_data_mnist_format
from src.plot import plot_sample_images
from src.Model import create_model, compile_and_train
from src.forward_passNN import my_sequential, sigmoid
import numpy as np

def main():
    try:
        logging.info("Starting main workflow")

        # Set your MNIST file paths here
        train_images = r"C:\Users\05she\Mnist\mnist_data\train-images.idx3-ubyte"
        train_labels = r"C:\Users\05she\Mnist\mnist_data\train-labels.idx1-ubyte"
        test_images = r"C:\Users\05she\Mnist\mnist_data\t10k-images.idx3-ubyte"
        test_labels = r"C:\Users\05she\Mnist\mnist_data\t10k-labels.idx1-ubyte"

        # Load data (limit for quick testing, remove num_train/num_test for full set)
        (X_train, y_train), (X_test, y_test) = load_data_mnist_format(
            train_images_path=train_images,
            train_labels_path=train_labels,
            test_images_path=test_images,
            test_labels_path=test_labels,
            num_train=1000,  # Use None for all data
            num_test=200
        )

        # Visualize some images
        plot_sample_images(X_train, y_train, num_images=64, img_shape=(28, 28))

        # Prepare labels for binary classification (e.g., digit 0 vs not-0)
        y_train_bin = (y_train == 0).astype(np.float32)
        y_test_bin = (y_test == 0).astype(np.float32)

        # Create and train model
        model = create_model(input_dim=784)
        model = compile_and_train(model, X_train, y_train_bin, epochs=10)

        # Example prediction
        pred = model.predict(X_test[0].reshape(1, 784))
        logging.info(f"Prediction for first test sample: {pred}, True label: {y_test_bin[0]}")

        # Numpy NN example (assuming you have weights W1, b1, W2, b2, W3, b3 loaded)
        # my_pred = my_sequential(X_test[0], W1, b1, W2, b2, W3, b3)
        # logging.info(f"Numpy NN prediction: {my_pred}")

    except Exception as e:
        logging.error(f"Unhandled exception in main workflow: {e}")
        raise CustomException(str(e), sys)

if __name__ == "__main__":
    main()