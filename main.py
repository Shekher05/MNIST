import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.logger import logging
from src.exception_h import CustomException
import os
import sys

from src.data import load_data_mnist_format
from src.Model import create_model, compile_and_train
import numpy as np
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("artifacts", "mnist_model.keras")

def load_and_predict(X_test, y_test):
    # Load the saved model
    model = load_model(MODEL_PATH)
    logging.info(f"loaded from the model {MODEL_PATH}")
    # Pick a random test sample
    idx = random.randint(0, len(X_test) - 1)
    sample_image = X_test[idx]
    true_label = y_test[idx]
    # Reshape and predict
    sample_pred = model.predict(sample_image.reshape(1, 784))
    predicted_label = np.argmax(sample_pred, axis=1)[0]
    # Print results
    print(f"True label: {true_label}, Predicted label: {predicted_label}")
    # Show image
    plt.imshow(sample_image.reshape(28, 28), cmap="gray")
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()


def main():
    try:
        logging.info("Starting main workflow")

        # Load data (paths default to ./mnist_data next to the repo)
        (X_train, y_train), (X_test, y_test) = load_data_mnist_format()
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")

        # Train only if no saved model exists; delete the file to retrain
        if not os.path.exists(MODEL_PATH):
            model = create_model(input_dim=784)
            model = compile_and_train(model, X_train, y_train, epochs=10)
            # Evaluate the model
            preds = model.predict(X_test)
            pred_classes = np.argmax(preds, axis=1)
            accuracy = np.mean(pred_classes == y_test)
            print(f"Test accuracy: {accuracy:.4f}")
            # Save the model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)
            logging.info(f"Model saved to {MODEL_PATH}")

         # Example prediction
        # sample_idx = 123  # Pick a random test example
        # sample_image = X_test[sample_idx]
        # sample_pred = model.predict(sample_image.reshape(1, 784))
        # predicted_digit = np.argmax(sample_pred)
        # true_digit = y_test[sample_idx]
        # print(f"Example prediction - True: {true_digit}, Predicted: {predicted_digit}")

        ## If you have created the model use the below code to load the model and predict
        load_and_predict(X_test, y_test)

        
    except CustomException:
        raise  # already logged and formatted where it was raised
    except Exception as e:
        logging.error(f"Unhandled exception in main workflow: {e}")
        raise CustomException(str(e), sys)
    
if __name__ == "__main__":
    main()