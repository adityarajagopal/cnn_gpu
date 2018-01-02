# https://www.tensorflow.org/tutorials/layers this website has the tutorial and explanations

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode) : 
    # the four arguments to size are [batch_size, width, height, num channels] 
    # the channels argument specifies 1 from grey scale and 3 for rgb 
    # the -1 calculates the correct size based on the other values which are set
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs = input_layer, 
        filters = 32, 
        kernel_size = [5,5], 
        padding="same", 
        activation=tf.nn.relu
    )
    # the output of this will produce an output of size [batch_size, 28, 28, 32] 
    # (one channel corresponding to each filter)

    pool1 = tf.layers.max_pooling2d(
        inputs = conv1, 
        pool_size=[2,2],
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs = pool1, 
        filters = 64, 
        kernel_size = [5,5], 
        padding = "same", 
        activation = tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        inputs = conv2, 
        pool_size = [2,2], 
        strides=2
    )

    # now we want to construct the dense layer to classify images 
    # for that we need to resize the feature map into the form [batch_size, num_features] 
    # pooling with 2,2 on a 28,28 will result in 7,7 output feature maps
    # conv2 had 64 filters, and hence we have 64 feature maps 
    # hence, the total number of features is 7*7*64
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    dense0 = tf.layers.dense (
        inputs = pool2_flat, 
        units = 1024, 
        activation = tf.nn.relu
    )
    # output of this is of size [batch_size, 1024] 

    # activation for this is linear (which is default) 
    dense1 = tf.layers.dense (
        inputs = dense0, 
        units = 10
    )   
    # the output here is [batch_size, 10] 
    # so for each of the 10, the predicted class is the max value along the rows 
    
    # add predictions if we are in the predicting mode
    predictions = {
        "classes" : tf.argmax(dense1, axis=1), 
        "probabilities" : tf.nn.softmax(dense1, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT : 
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # creation of onehot_labels from the labels input 
    # to pass into the softmax_cross_entropy loss function
    onehot_labels = tf.one_hot(
        indices = tf.cast(labels,tf.int32),
        depth = 10
    )
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels, 
        logits = dense1
    )

    # if mode is training, setup the gradient descent optimiser 
    # with learning rate 0.001
    if mode == tf.estimator.ModeKeys.TRAIN : 
        optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimiser.minimize (
            loss = loss, 
            global_step = tf.train.get_global_step()
        )
        
        return tf.estimator.EstimatorSpec(
            mode = mode, 
            loss = loss, 
            train_op = train_op
        )

    # add error values to check learning
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, 
        loss=loss, 
        eval_metric_ops=eval_metric_ops
    )

def main(unused_argv) : 
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images 
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images 
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # model_fn takes in function which defines what to do during 
    # TRAIN, EVAL, and PREDICT stages
    mnist_classifier = tf.estimator.Estimator (
        model_fn = cnn_model_fn, 
        model_dir = "tmp/mnist_convnet_model"
    )
    
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook (
        # tensors = tensors_to_log, 
        every_n_iter = 50
    )

    # train model 
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : train_data}, 
        y = train_labels, 
        batch_size = 100, 
        num_epochs = None, 
        shuffle = True
    )
    mnist_classifier.train(
        input_fn = train_input_fn, 
        steps = 20000, 
        hooks = [logging_hook]
    )

    # evaluate model 
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x" : eval_data},
        y = eval_labels, 
        num_epochs = 1, 
        shuffle = False
    )
    eval_results = mnist_classifier.evaluate(input_fn = eval_input_fn)

if __name__ == "__main__" : 
    tf.app.run()
