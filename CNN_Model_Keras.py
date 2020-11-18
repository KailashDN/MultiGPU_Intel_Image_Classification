from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pre_process_data as pre_data
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# Here I set it default to relu so that our variable became of that type
act_func_global = tf.nn.relu
print(act_func_global)
is_Swish = False
print('is a Swish function: ',is_Swish)
is_SwishBeta = False
print('is a Swish Beta function: ',is_SwishBeta)
is_logging = False
print('is a first function: ',is_Swish)
b_size=32
print('batch size: ', b_size)


def check_available_GPUS():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)
    print(f'{gpu_num} GPUs are detected : {gpu_names}')
    return gpu_num


def execution_time(model_start_time,model_end_time):
    print('Model execution start Time:',round(model_start_time,0))
    print('Model execution end Time:',round(model_end_time,0))
    excn_time= model_end_time - model_start_time
    print('Model execution Time:',round(excn_time/60,2),'minutes')


def swish_1(x):
    global act_func_global
    act_func_global = x*tf.nn.sigmoid(x)
    return act_func_global


def swish_beta(x):
    beta=tf.Variable(initial_value=0.8, trainable=True, name='swish_beta')
    print('swish_beta value: ', beta)
    # trainable parameter beta
    global act_func_global
    act_func_global = x*tf.nn.sigmoid(beta*x)
    return act_func_global


def cnn_model_fn():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    batch_size = 32
    num_val_samples = 10000

    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    # x_train, y_train, x_test, y_test = pre_data.pre_process()

    # # Preprocess the data (these are Numpy arrays)
    # x_train = x_train.reshape(-1, 150, 150, 3).astype("float32") / 255
    # x_test = x_test.reshape(-1, 150, 150, 3).astype("float32") / 255
    # y_train = y_train.astype("float32")
    # y_test = y_test.astype("float32")
    # Return the MNIST dataset in the form of a `tf.data.Dataset`.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


def set_GPU_Strategy(num_GPU):
    print("Available GPUs: ", tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_GPU)
    # TF 2.0
    strategy = tf.distribute.MirroredStrategy()
    # run_config = tf.estimator.RunConfig(train_distribute=strategy)
    return strategy


def model_pipeline(act_function, Islog:False, num_GPU):
    # Set up logging for predictions
    dict_act = {'sigmoid': tf.nn.sigmoid, 'tanh':tf.nn.tanh, 'relu':tf.nn.relu, 'leaky_relu':tf.nn.leaky_relu,
                'swish':tf.nn.sigmoid, 'swish_beta':tf.nn.sigmoid}

    if act_function == 'swish':
        global is_Swish
        is_Swish = True
    elif act_function == 'swish_beta':
        global is_SwishBeta
        is_SwishBeta = True

    global is_logging
    is_logging = Islog
    # act_func_global = tf.nn.sigmoid
    global act_func_global
    act_func_global = dict_act[act_function]

    # ----------------- Model Run ---------------
    model_start_time = time.time()

    run_config_strategy = set_GPU_Strategy(num_GPU)

    # Open a strategy scope.
    with run_config_strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = cnn_model_fn()

    # Train the model on all available devices.
    train_dataset, val_dataset, test_dataset = get_dataset()
    model.fit(train_dataset, epochs=50, validation_data=val_dataset)

    # Test the model on all available devices.
    model.evaluate(test_dataset)

    model_end_time = time.time()
    execution_time(model_start_time, model_end_time)


model_pipeline('relu', True, 1)
