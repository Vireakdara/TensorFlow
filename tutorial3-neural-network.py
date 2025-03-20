import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices('GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load dataset from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()












