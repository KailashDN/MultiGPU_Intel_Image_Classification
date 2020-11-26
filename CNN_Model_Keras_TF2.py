from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import pre_process_data as pre_data
from keras.callbacks import ModelCheckpoint
import tensorflow_datasets as tfds
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import time
import os
import logging
tf.get_logger().setLevel(logging.ERROR)

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 3 no TF debug info
tf.debugging.set_log_device_placement(True)
#tf.config.experimental.list_physical_devices('GPU')

# Here I set it default to relu so that our variable became of that type
act_func_global = tf.nn.relu
print(act_func_global)
is_Swish = False
print('is a Swish function: ',is_Swish)
is_SwishBeta = False
print('is a Swish Beta function: ',is_SwishBeta)
is_logging = False
print('is a first function: ',is_Swish)
b_size=256
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


def act_func(x):
    dict_act = {'sigmoid': tf.nn.sigmoid(x), 'tanh': tf.nn.tanh(x), 'relu': tf.nn.relu(x), 'leaky_relu': tf.nn.leaky_relu(x),
                'swish': tf.nn.sigmoid(x), 'swish_beta': tf.nn.sigmoid(x)}
    if act_func_global == 'swish':
        # act_func_var = x*tf.nn.sigmoid(x)
        return x * keras.backend.sigmoid(x)

    if act_func_global == 'swish_beta':
        beta = 1.5  # 1, 1.5 or 2
        return x * keras.backend.sigmoid(beta * x)
    else:
        return dict_act[act_func_global]


def set_GPU_Strategy(num_GPU):
    gpu_list = ['/gpu:'+str(i) for i in range(0, num_GPU) ]
    print("Available GPUs: ", tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, False)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_GPU)
    # TF 2.0
    print('GPUs will be used in training:', gpu_list)
    strategy = tf.distribute.MirroredStrategy(gpu_list)
    # run_config = tf.estimator.RunConfig(train_distribute=strategy)
    return strategy


def model_pipeline(act_function, Islog:False, num_GPU):

    global act_func_global
    act_func_global = act_function
    print('----------------- Model Run ---------------')
    # ----------------- Model Run ---------------
    model_start_time = time.time()
    #check_available_GPUS()
    run_config_strategy = set_GPU_Strategy(num_GPU)
    print('GPU Run Strategy:',run_config_strategy)

    print('----------------- GET DATA -----------------')
    # ----------------- Dataset -----------------
    train_data, train_labels, eval_data, eval_labels = pre_data.pre_process()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).cache().batch(b_size)
    eval_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).cache().batch(b_size)

    # Open a strategy scope. Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`
    print('----------------- Model Create ---------------')
    # ----------------- Model Run ---------------
    with run_config_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation=act_func, input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.BatchNormalization(epsilon=0.001),
            tf.keras.layers.Conv2D(64, 3, activation=act_func),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
            tf.keras.layers.LayerNormalization(epsilon=0.001),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=act_func),
            tf.keras.layers.Dense(6)
        ])

        # model = tf.keras.Sequential()
        # # 1st convolution layer
        # model.add(tf.keras.layers.Conv2D(32, (3, 3),
        #                                  activation=act_func,
        #                                  input_shape=(64, 64, 3)))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        # model.add(tf.keras.layers.LayerNormalization(epsilon=0.001))
        # # 2nd convolution layer
        # model.add(tf.keras.layers.Conv2D(64, (3, 3),
        #                                  activation=act_func))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1))
        # model.add(tf.keras.layers.BatchNormalization(epsilon=0.001))
        #
        # model.add(tf.keras.layers.Flatten())
        # # Fully connected layer. 1 hidden layer consisting of 512 nodes
        # model.add(tf.keras.layers.Dense(128, activation=act_func))
        # model.add(tf.keras.layers.Dense(6, activation='softmax'))

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    print("checkpoint_prefix: ", checkpoint_prefix)

    # Function for decaying the learning rate.
    # You can define any decay function you need.
    def decay(epoch):
        if epoch < 3:
            return 1e-3
        elif 3 <= epoch < 7:
            return 1e-4
        else:
            return 1e-5

    # Callback for printing the LR at the end of each epoch.
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))

    callbacks = [
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
        #                                    save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]
    print('------------------- train ------------------- ')
    model.fit(train_dataset, epochs=12, callbacks=callbacks)

    print('------------------- Evaluate ------------------- ')
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    eval_loss, eval_acc = model.evaluate(eval_dataset)
    print('\n\nEval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
    print('Number of GPUs:', num_GPU)
    model_end_time = time.time()
    execution_time(model_start_time, model_end_time)

""" Parameters: activation function, logging op, Number of GPU(Discovery 4 p100 available) """
model_pipeline('relu', True, 4)
