import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Initialization
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
print(x)

x = tf.constant([[1, 2, 3], [4, 5, 6]], shape=(2, 3))
print(x)

x = tf.eye(3)
print(x)

x = tf.ones((4, 3))
print(x)

x = tf.zeros((3, 2, 5))
print(x)

x = tf.random.uniform((2, 2), minval=0, maxval=1)
print(x)
