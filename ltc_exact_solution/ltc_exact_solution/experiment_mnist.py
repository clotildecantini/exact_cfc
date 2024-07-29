"""This file aims to train an network with LTC layer."""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow import keras
from sklearn.model_selection import train_test_split
from ltc_exact_solution.model_cells import ODELayer, ApproxLTCLayer, ExactLTCLayer

def load_mnist(batch_size : int = 32):
    """This function aims to load the MNIST dataset.
    Returns:
    train_dataset: tf.data.Dataset: The training dataset.
    val_dataset: tf.data.Dataset: The validation dataset.
    test_dataset: tf.data.Dataset: The test dataset."""

    # Load Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype('float32') / 255

    # Create train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Batch and prefetch for performance
    batch_size = batch_size
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset

def create_model(ltc_layer: str = 'exact', units: int = 64, omega: float = 0.1, 
                 number_of_classes: int = 10, input_shape: tuple = (28, 28, 1)):
    """This function aims to create the model with the LTC layer."""
    if ltc_layer == 'ode':
        ltc_layer = ODELayer(units=units, omega=omega)
    elif ltc_layer == 'approx':
        ltc_layer = ApproxLTCLayer(units=units, omega=omega)
    else:
        ltc_layer = ExactLTCLayer(units=units, omega=omega)

    # Define the model
    inputs = keras.Input(shape=input_shape)
    ltc = ltc_layer(inputs)
    
    # Add BatchNormalization layer
    normalized = keras.layers.BatchNormalization()(ltc)
    
    flattened = keras.layers.Flatten()(normalized)
    outputs = keras.layers.Dense(number_of_classes, activation='softmax')(flattened)
    
    model_ltc = keras.Model(inputs, outputs)

    # Compile the model
    model_ltc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model_ltc