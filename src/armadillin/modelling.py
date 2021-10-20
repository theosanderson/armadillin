import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LocallyConnected1D, LocallyConnected2D

from keras import backend as K
from tensorflow.keras import initializers
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from . import input
import numpy as np
import h5py
import gzip


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))




prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude




class MultiplyByWeights(keras.layers.Layer):
    def __init__(self, weight_reg_value=0.01, **kwargs):
        super(MultiplyByWeights, self).__init__()
        self.weight_reg_value = weight_reg_value

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1]),
                                 initializer=initializers.Ones(),
                                 trainable=True,
                                 regularizer=keras.regularizers.l1(
                                     self.weight_reg_value))
        self.the_input_shape = input_shape

    def get_config(self):
        config = super(MultiplyByWeights, self).get_config()
        config['weight_reg_value'] = self.weight_reg_value
        return config

    def get_prunable_weights(self):
        return self.weights

    def call(self, inputs):

        reshaped_weights = K.reshape(self.w, self.the_input_shape[1:])
        return tf.math.multiply(inputs, reshaped_weights)


def build_model(config):

    if config['mode'] == "pretrained_pruned":
        input = tf.keras.Input(shape=(config['features_length']),
                           dtype=tf.float32,
                           name="input")

    else:
        input = tf.keras.Input(shape=(config['seq_length'],
                                  len(config['alphabet'])),
                           dtype=tf.float32,
                           name="input")


    x = Flatten(name="flatten")(input)
    x = MultiplyByWeights(weight_reg_value=1e-8, name="local_1")(x)


    x = Dense(
        500,
        activation='relu',
        name="initial_dense",
    )(x)
    x = Dense(500, activation='relu', name="second_dense")(x)
    x = Dense(len(config['all_lineages']),
              activation='sigmoid',
              name="output_dense")(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    return model


def build_pruning_model(config, model = None, pruning_params = None):
    if model is None:
        model = build_model(config)

    

    def prune_local(layer):
        if isinstance(layer, MultiplyByWeights):
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer, **pruning_params)
        return layer

    model = tf.keras.models.clone_model(
        model,
        clone_function=prune_local,
    )
    return model


def load_saved_model(filename):
    if filename.endswith(".h5.gz"):
        f = gzip.open(filename, 'rb')
        target = h5py.File(f)
    else:
        target = filename
        

    with tfmot.sparsity.keras.prune_scope():
        model = tf.keras.models.load_model(target,
                                           custom_objects={
                                               "f1_m":
                                               f1_m,
                                               "precision_m":
                                               precision_m,
                                               "recall_m":
                                               recall_m,
                                               "MultiplyByWeights":
                                               MultiplyByWeights
                                           }, compile=False)
        return model


def compile_model(model, learning_rate):
    opt = tf.keras.optimizers.Adam(lr=learning_rate, )
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['acc', f1_m, precision_m, recall_m, 'binary_crossentropy'])


def create_pretrained_pruned_model(initial_model):


    masking_weights = initial_model.get_layer(
        "prune_low_magnitude_multiply_by_weights").get_weights()
    remaining_positions = np.where(masking_weights[0] != 0)[0]

    model_config = {
        "alphabet": input.alphabet,
        "all_lineages": input.all_lineages,
        "features_length": remaining_positions.shape[0],
        "mode": "pretrained_pruned"
    }
    new_model = build_model(model_config)

    old_initial_weights = initial_model.get_layer(
        "initial_dense").get_weights()
    #print(old_initial_weights[0].shape)
    assert old_initial_weights[0].shape == (29891 * 5, 500)
    filtered = old_initial_weights[0][remaining_positions, :]
    new_initial_weights = old_initial_weights
    new_initial_weights[0] = filtered

    print(initial_model.summary())
    for layer in initial_model.layers:
        print(layer.name)

    old_multipliers = masking_weights
    new_multipliers = old_multipliers
    new_multipliers[0] = old_multipliers[0][remaining_positions]

    print(new_model.summary())


    for layer in new_model.layers:
        if layer.name == "initial_dense":
            layer.set_weights(new_initial_weights)
        elif layer.name == "multiply_by_weights_1":
            layer.set_weights(new_multipliers)
        else:
            layer.set_weights(
                initial_model.get_layer(layer.name).get_weights())
    return new_model, remaining_positions
