from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pre_process_data as pre_data
import os
# tf.get_logger().setLevel('WARNING')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')

# Here I set it default to relu so that our variable became of that type
act_func_global = tf.nn.relu
print(act_func_global)
is_Swish = False
print('is a Swish function: ',is_Swish)
is_SwishBeta = False
print('is a Swish Beta function: ',is_SwishBeta)
is_logging = False
print('is a first function: ',is_Swish)
b_size=128
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


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # act_func=tf.nn.relu

    print('act_func_global: ', act_func_global)

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])  # All of our images are of size 150*150
    if is_logging == True:
        print('input layer shape: ', tf.shape(input_layer))
        print(input_layer.get_shape())
    if is_Swish == True:
        swish_1(input_layer)
    if is_SwishBeta == True:
        swish_beta(input_layer)
    # Convolutional Layer #1
    conv1 = tf.compat.v1.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        use_bias=True,
        activation=act_func_global)
    if is_logging == True:
        print('con1 shape: ', tf.shape(conv1))
        print(conv1.get_shape())

    # Pooling Layer #1
    pool1 = tf.compat.v1.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    if is_logging == True:
        print('pool1 Layer shape: ', tf.shape(pool1))
        print(pool1.get_shape())

    # local_response_normalization
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 3.0, beta=0.75, name='norm1')
    if is_logging == True:
        print('local_response_normalization shape: ', tf.shape(norm1))
        print(norm1.get_shape())

    if is_Swish == True:
        swish_1(norm1)
    if is_SwishBeta == True:
        swish_beta(norm1)
    # Convolutional Layer #2
    conv2 = tf.compat.v1.layers.conv2d(
        inputs=norm1,
        filters=64,
        kernel_size=[3, 3],
        padding="valid",
        activation=act_func_global)
    if is_logging == True:
        print('con2 Layer shape: ', tf.shape(conv2))
        print(conv2.get_shape())

    # Pooling Layer  # 2
    pool2 = tf.compat.v1.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    if is_logging == True:
        print('pool2 Layer shape: ', tf.shape(pool2))
        print(pool2.get_shape())

    # We'll flatten our feature map (pool2) to shape [batch_size, features], so that our tensor has only two dimensions
    # pool2_flat = tf.reshape(pool2, [-1, 37 * 37 * 64])
    pool2_flat = tf.reshape(pool2, [-1, pool2.get_shape()[1] * pool2.get_shape()[2] * 64])
    if is_logging == True:
        print('pool2_flat shape: ', tf.shape(pool2_flat))
        print(pool2_flat.get_shape())

    if is_Swish == True:
        swish_1(pool2_flat)

    if is_SwishBeta == True:
        swish_beta(pool2_flat)
    # Dense Layer
    dense = tf.compat.v1.layers.dense(inputs=pool2_flat, units=256, activation=act_func_global)
    if is_logging == True:
        print('dense Layer shape: ', tf.shape(dense))
        print(dense.get_shape())

    # Drop the 10% of the input
    #     dropout = tf.layers.dropout(inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)
    #     print('dropout Regularizatin shape: ',tf.shape(dropout))
    #     print(dropout.get_shape())

    # Logits Layer
    logits = tf.compat.v1.layers.dense(inputs=dense, units=6)  # Here we have 6 Classes
    if is_logging == True:
        print('logits Layer shape: ', tf.shape(logits))
        print(logits.get_shape())

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=1.0)
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.AUTO)(labels, logits)
    print('loss: ', loss, '\nvar_list: ', tf.compat.v1.train.get_or_create_global_step())
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # optimizer = tf.optimizers.Adam(learning_rate=0.01, beta_1=0.8, beta_2=0.999, epsilon=1e-07)
        print("Optimizer: ", optimizer)
        # train_op = optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step())
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        # print('train_op: ',train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    print('Evaluation: ')
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def set_GPU_Strategy(num_GPU):
    print("Available GPUs: ", tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=num_GPU)
    # TF 2.0
    strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(train_distribute=strategy)
    return run_config, strategy


# Create the Estimator
def estimator_path(function_name, run_config):
    actfn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, config=run_config,
            model_dir="Estimator_"+function_name+"//actfn_convnet_model")
    return actfn_classifier


def input_fn(images, labels, epochs, batch_size):
    # Convert the inputs to a Dataset. (E)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # Shuffle, repeat, and batch the examples. (T)
    SHUFFLE_SIZE = 500
    ds = ds.shuffle(SHUFFLE_SIZE).repeat(epochs).batch(batch_size)
    ds = ds.prefetch(2)
    # Return the dataset. (L)
    return ds


def train_model(actfn_classifier, logging_hook, train_data, train_labels):
    # Train the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    # TF 2.0
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=b_size,
        num_epochs=None,
        shuffle=True)

    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).cache().repeat(100).batch(128)
    # train one step and display the probabilities
    print("train_input_fn: ", train_input_fn)
    print("dataset: ", dataset)

    actfn_classifier.train(dataset, steps=1, hooks=[logging_hook])
    # Without logging
    actfn_classifier.train(input_fn=train_input_fn, steps=1000)


def evaluate_model(actfn_classifier, eval_data, eval_labels):
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=20,
        shuffle=False)

    eval_results = actfn_classifier.evaluate(input_fn=eval_input_fn)
    return eval_results


def set_globals(act_function, Islog):
    dict_act = {'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh, 'relu': tf.nn.relu, 'leaky_relu': tf.nn.leaky_relu,
                'swish': tf.nn.sigmoid, 'swish_beta': tf.nn.sigmoid}

    if act_function == 'swish':
        global is_Swish
        is_Swish = True
    elif act_function == 'swish_beta':
        global is_SwishBeta
        is_SwishBeta = True

    global is_logging
    is_logging = Islog
    global act_func_global
    act_func_global = dict_act[act_function]


def model_pipeline(act_function, Islog:False, num_GPU):
    set_globals(act_function, Islog)
    print("check_available_GPUS: ", check_available_GPUS())
    # train_data, train_labels, eval_data, eval_labels = pre_data.pre_process()

    tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)


    # Model Run
    model_start_time = time.time()

    run_config, strategy = set_GPU_Strategy(num_GPU)
    act_fn_classifier = estimator_path(act_function, run_config)

    train_data, train_labels, eval_data, eval_labels = pre_data.pre_process()
    # TRAINING_SIZE = len(train_data)
    # TEST_SIZE = len(eval_data)
    # train_data = np.asarray(train_data, dtype=np.float32) / 255
    # train_data = train_data.reshape((TRAINING_SIZE, 64, 64, 3))
    #
    # eval_data = np.asarray(eval_data, dtype=np.float32) / 255
    # eval_data = eval_data.reshape((TEST_SIZE, 64, 64, 3))
    #
    # LABEL_DIMENSIONS = 6
    # train_labels = tf.keras.utils.to_categorical(train_labels, LABEL_DIMENSIONS)
    # eval_labels = tf.keras.utils.to_categorical(eval_labels, LABEL_DIMENSIONS)
    # train_labels = train_labels.astype(np.float32)
    # eval_labels = eval_labels.astype(np.float32)

    # BUFFER_SIZE = 10000
    # BATCH_SIZE_PER_REPLICA = 64
    # BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    # def scale(image, label):
    #     image = tf.cast(image, tf.float32)
    #     image /= 255
    #
    #     return image, label
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).cache().batch(128)
    # eval_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).cache().batch(128)

    BATCH_SIZE = 64
    EPOCHS = 5
    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=b_size,
        num_epochs=None,
        shuffle=True)
    # act_fn_classifier.train(lambda: train_input_fn(train_data,
    #                             train_labels,
    #                             epochs=EPOCHS,
    #                             batch_size=BATCH_SIZE), steps=1, hooks=[logging_hook])
    act_fn_classifier.train(input_fn=lambda: train_input_fn, steps=1, hooks=[logging_hook])
    # Without logging
    act_fn_classifier.train(input_fn=input_fn, steps=1000)
    # train_model(actfn_classifier, logging_hook, train_data, train_labels)

    # eval_results = evaluate_model(eval_dataset, eval_data, eval_labels)
    print(f"======== Evaluation Results of {act_function}========\n  eval_results")

    model_end_time = time.time()
    execution_time(model_start_time, model_end_time)


model_pipeline('relu', True, 1)
