import numpy as np
from src.logger import logging
from src.exception_h import CustomException
import sys

def sigmoid(x):
    try:
        logging.debug("Applying sigmoid activation")
        return 1. / (1. + np.exp(-x))
    except Exception as e:
        logging.error(f"Error in sigmoid: {e}")
        raise CustomException(str(e), sys)

def my_dense(a_in, W, b, g):
    try:
        logging.info("Running my_dense layer")
        z = np.matmul(a_in, W) + b
        a_out = g(z)
        return a_out
    except Exception as e:
        logging.error(f"Error in my_dense: {e}")
        raise CustomException(str(e), sys)

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    try:
        logging.info("Running my_sequential forward pass")
        a1 = my_dense(x, W1, b1, sigmoid)
        a2 = my_dense(a1, W2, b2, sigmoid)
        a3 = my_dense(a2, W3, b3, sigmoid)
        return a3
    except Exception as e:
        logging.error(f"Error in my_sequential: {e}")
        raise CustomException(str(e), sys)

def my_dense_v(A_in, W, b, g):
    try:
        logging.info("Running my_dense_v layer (vectorized)")
        z = np.matmul(A_in, W) + b
        A_out = g(z)
        return A_out
    except Exception as e:
        logging.error(f"Error in my_dense_v: {e}")
        raise CustomException(str(e), sys)

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    try:
        logging.info("Running my_sequential_v forward pass (vectorized)")
        A1 = my_dense_v(X, W1, b1, sigmoid)
        A2 = my_dense_v(A1, W2, b2, sigmoid)
        A3 = my_dense_v(A2, W3, b3, sigmoid)
        return A3
    except Exception as e:
        logging.error(f"Error in my_sequential_v: {e}")
        raise CustomException(str(e), sys)