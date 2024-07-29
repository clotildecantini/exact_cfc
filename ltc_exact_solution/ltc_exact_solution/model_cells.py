"""This file aims to define the layers of the model with the ODE layer, the approximate layer and the exact layer."""

import tensorflow as tf
from tensorflow import keras

class ODELayer(keras.layers.Layer):
    def __init__(self, units, omega, dt=0.1, ode_unfolds=3, **kwargs):
        super(ODELayer, self).__init__(**kwargs)
        self.units = units
        self.omega = omega
        self.dt = dt
        self.ode_unfolds = ode_unfolds

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(self.units, self.input_dim),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='A')
        self.sigma = self.add_weight(shape=(self.units, self.input_dim),
                                     initializer='random_normal',
                                     trainable=True,
                                     name='sigma')
        self.mu = self.add_weight(shape=(self.units, self.input_dim),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='mu')
        self.x0 = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  name='x0')

    def call(self, inputs):
        inputs_reshaped = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]])

        def step(x_prev, inputs_t):
            x = x_prev
            dt_unfold = self.dt / self.ode_unfolds
            for _ in range(self.ode_unfolds):
                f = tf.math.sigmoid(self.sigma * (tf.expand_dims(inputs_t, 1) - self.mu))
                dx = -self.omega * x + tf.reduce_sum(f * (self.A - tf.expand_dims(x, -1)), axis=-1)
                x = x + dt_unfold * dx
            return x

        initial_state = tf.tile(self.x0[tf.newaxis, :], [tf.shape(inputs_reshaped)[0], 1])
        outputs = tf.scan(step, tf.transpose(inputs_reshaped, [1, 0, 2]), initializer=initial_state)
        return tf.transpose(outputs, [1, 0, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2], self.units)

class ApproxLTCLayer(keras.layers.Layer):
    def __init__(self, units, omega, **kwargs):
        super(ApproxLTCLayer, self).__init__(**kwargs)
        self.units = units
        self.omega = omega

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(self.units, self.input_dim),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='A')
        self.sigma = self.add_weight(shape=(self.units, self.input_dim),
                                     initializer='random_normal',
                                     trainable=True,
                                     name='sigma')
        self.mu = self.add_weight(shape=(self.units, self.input_dim),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='mu')
        self.x0 = self.add_weight(shape=(self.units,),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='x0')

    def call(self, inputs):
        inputs_reshaped = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]])
        
        def step(x_prev, inputs_t):
            t = tf.cast(tf.range(tf.shape(inputs_t)[0]), dtype=tf.float32)
            I_t = tf.expand_dims(inputs_t, 1)
            
            f_I_t = tf.math.sigmoid(self.sigma * (I_t - self.mu))
            f_neg_I_t = tf.math.sigmoid(-self.sigma * (I_t - self.mu))
            
            exp_term = tf.exp(-self.omega * t[:, tf.newaxis, tf.newaxis] - f_I_t*t[:, tf.newaxis, tf.newaxis])
            
            x = (self.x0[:, tf.newaxis] - self.A) * exp_term * f_neg_I_t + self.A
            return tf.reduce_sum(x, axis=-1)

        initial_state = tf.tile(self.x0[tf.newaxis, :], [tf.shape(inputs_reshaped)[0], 1])
        outputs = tf.scan(step, tf.transpose(inputs_reshaped, [1, 0, 2]), initializer=initial_state)
        return tf.transpose(outputs, [1, 0, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2], self.units)

class ExactLTCLayer(keras.layers.Layer):
    def __init__(self, units, omega, **kwargs):
        super(ExactLTCLayer, self).__init__(**kwargs)
        self.units = units
        self.omega = omega

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.A = self.add_weight(shape=(self.units, self.input_dim),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='A')
        self.sigma = self.add_weight(shape=(self.units, self.input_dim),
                                     initializer='random_normal',
                                     trainable=True,
                                     name='sigma')
        self.mu = self.add_weight(shape=(self.units, self.input_dim),
                                  initializer='random_normal',
                                  trainable=True,
                                  name='mu')
        self.x0 = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  trainable=True,
                                  name='x0')

    def call(self, inputs):
        # Reshape inputs to (batch_size, time_steps, features)
        inputs_reshaped = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2], inputs.shape[3]])

        def step(x_prev, inputs_t):
            f = tf.math.sigmoid(self.sigma * (tf.expand_dims(inputs_t, 1) - self.mu))
            u = f / (self.omega + tf.reduce_sum(f, axis=-1, keepdims=True))
            x = tf.exp(-self.omega - tf.reduce_sum(f, axis=-1)) * (x_prev - tf.reduce_sum(self.A * u, axis=-1)) + tf.reduce_sum(self.A * u, axis=-1)
            return x

        initial_state = tf.tile(self.x0[tf.newaxis, :], [tf.shape(inputs_reshaped)[0], 1])
        outputs = tf.scan(step, tf.transpose(inputs_reshaped, [1, 0, 2]), initializer=initial_state)
        return tf.transpose(outputs, [1, 0, 2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[2], self.units)