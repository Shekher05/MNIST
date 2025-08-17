import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from logger import logging
from exception_h import CustomException
import sys

def create_model(input_dim=784):
    try:
        logging.info("Creating Keras Sequential model")
        model = Sequential([
            tf.keras.Input(shape=(input_dim,)),
            Dense(25, activation='sigmoid', name='layer1'),
            Dense(15, activation='sigmoid', name='layer2'),
            Dense(1, activation='sigmoid', name='layer3')
        ], name="my_model")
        logging.info("Model created")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        raise CustomException(str(e), sys)

def compile_and_train(model, X, y, epochs=20):
    try:
        logging.info("Compiling and training model")
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy']
        )
        model.fit(X, y, epochs=epochs)
        logging.info("Model training complete")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise CustomException(str(e), sys)