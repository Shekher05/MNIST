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

        # Load data
        (X_train, y_train), (X_test, y_test) = load_data_mnist_format(
            train_images_path=train_images,
            train_labels_path=train_labels,
            test_images_path=test_images,
            test_labels_path=test_labels,
            num_train=None,
            num_test=None
        )
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")

        # Visualize some images
        # plot_sample_images(X_train, y_train, num_images=100, img_shape=(28, 28))

        # For multi-class, we use the labels directly (no binary conversion)
        # y_train and y_test are already 0-9 labels

        # Create and train model
        model = create_model(input_dim=784)
        model = compile_and_train(model, X_train, y_train, epochs=10)

        # Evaluate the model
        preds = model.predict(X_test)
        pred_classes = np.argmax(preds, axis=1)
        accuracy = np.mean(pred_classes == y_test)
        print(f"Test accuracy: {accuracy:.4f}")

        # Save the model
        import os
        model_save_path = os.path.join("artifacts", "mnist_model.keras")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logging.info(f"Model saved to {model_save_path}")

        # Example prediction
        sample_idx = 123  # Pick a random test example
        sample_image = X_test[sample_idx]
        sample_pred = model.predict(sample_image.reshape(1, 784))
        predicted_digit = np.argmax(sample_pred)
        true_digit = y_test[sample_idx]
        print(f"Example prediction - True: {true_digit}, Predicted: {predicted_digit}")
        
    except Exception as e:
        logging.error(f"Unhandled exception in main workflow: {e}")
        raise CustomException(str(e), sys)
    
if __name__ == "__main__":
    main()