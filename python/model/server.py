import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import numpy as np
tf.get_logger().setLevel('ERROR')
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


epochs = 5
batch_size = 64
l2_norm_clip = 1.0
noise_multiplier = 0.3
num_microbatches = 1
learning_rate = 0.25

optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

loss_categorical = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_binary = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class Server:
    def __init__(self, dataset, method):
        self.dataset = dataset
        self.method = method
        self.model = self.get_model(dataset, method)

    def get_model(self, dataset, method):
        model = ''
        if dataset == 'minst':
            model = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])
        elif dataset == 'credit card':
                model = Sequential([
                Flatten(input_shape=(29,)),  # Flatten the 28x28 images into a 1D array
                Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
                Dense(1, activation='sigmoid')  # Output layer with 10 units for 10 classes (digits 0-9) and softmax activation
            ])
        else:
            pass
        if method != "differential privacy":
            if dataset != 'minst':
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            if dataset == 'minst':
                model.compile(optimizer=optimizer, loss=loss_categorical, metrics=['accuracy'])
            else:
                model.compile(optimizer=optimizer, loss=loss_binary, metrics=['accuracy'])
        return model