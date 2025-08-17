import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.logger import logging
from src.exception_h import CustomException
import sys

def create_model(input_dim=784):
    try:
        logging.info("Creating Keras Sequential model for multi-class digit recognition")
        model = Sequential([
            tf.keras.Input(shape=(input_dim,)),
            Dense(128, activation='relu', name='layer1'),
            Dense(64, activation='relu', name='layer2'),
            Dense(10, activation='softmax', name='output')  # 10 outputs for digits 0-9
        ], name="mnist_model")
        logging.info("Model created")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        raise CustomException(str(e), sys)

def compile_and_train(model, X, y, epochs=30):
    try:
        logging.info("Compiling and training model")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Changed for multi-class
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )
        model.fit(X, y, epochs=epochs)
        logging.info("Model training complete")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise CustomException(str(e), sys)