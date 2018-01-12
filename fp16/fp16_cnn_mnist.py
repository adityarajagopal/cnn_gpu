from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

tf.logging.set_verbosity(tf.logging.INFO)

def float32_variable_storage_getter (
        getter, name, shape=None, dtype=None,
        initializer=None, regularizer=None,
        trainable=True,
        *args, **kwargs
) :
    """Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def gradients_with_loss_scaling(loss, variables, loss_scale):
    """Gradient calculation with loss scaling to improve numerical stability
    when training with float16.
    """
    return [grad / loss_scale for grad in tf.gradients(loss * loss_scale, variables)]

def variables_lenet5 (dtype) :
    # window size is 5x5
    # data depth is 1 (num channels)
    # 6 output feature maps 
    w1 = tf.get_variable('w1', [5,5,1,6], dtype)
    b1 = tf.get_variable('b1', [6], dtype, initializer=tf.zeros_initializer())

    # window size is 5x5
    # 6 outpute feature maps from before mean data depth of 6
    # 16 output feature maps 
    w2 = tf.get_variable('w2', [5,5,6,16], dtype)
    b2 = tf.get_variable('b2', [16], dtype, initializer=tf.zeros_initializer())

    # fully connected layer - input 16 feature maps, each 5x5
    # fully connected layer - output 120 features
    init = tf.constant_initializer(np.full((1,120), 1.0, dtype=np.float16))
    w3 = tf.get_variable('w3', [5*5*16, 120], dtype)
    b3 = tf.get_variable('b3', [120], dtype, initializer=init)
    
    # fully connected layer - input 120
    # fully connected layer - output 84
    init = tf.constant_initializer(np.full((1,84), 1.0, dtype=np.float16))
    w4 = tf.get_variable('w4', [120,84], dtype)
    b4 = tf.get_variable('b4', [84], dtype, initializer=init)

    # classification fc layer - input 84 
    # classification fc layer- output 10 (1 per digit in case of MNIST)
    init = tf.constant_initializer(np.full((1,10), 1.0, dtype=np.float16))
    w5 = tf.get_variable('w5', [84,10], dtype)
    b5 = tf.get_variable('b5', [10], dtype, initializer=init)

    return  {'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4, 'w5':w5, 
             'b1':b1, 'b2':b2, 'b3':b3, 'b4':b4, 'b5':b5}

def model_lenet5 (dtype, trainVar, nbatch) :
    data = tf.placeholder (dtype, shape=(nbatch,28,28,1))

    layer1_conv = tf.nn.conv2d (
        data, 
        trainVar['w1'], 
        [1,1,1,1], 
        padding = 'SAME'
    )
    layer1_actv = tf.sigmoid (layer1_conv + trainVar['b1'])
    layer1_pool = tf.nn.avg_pool (
        value=layer1_actv, 
        ksize=[1,2,2,1], 
        strides=[1,2,2,1], 
        padding='VALID'
    )
    
    layer2_conv = tf.nn.conv2d (
        layer1_pool, 
        trainVar['w2'], 
        strides=[1,1,1,1], 
        padding = 'VALID'
    )
    layer2_actv = tf.sigmoid (layer2_conv + trainVar['b2'])
    layer2_pool = tf.nn.avg_pool (
        value=layer2_actv, 
        ksize=[1,2,2,1], 
        strides=[1,2,2,1], 
        padding='VALID'
    )
    
    flat_layer = tf.contrib.layers.flatten (layer2_pool)
    layer3_fc = tf.matmul (flat_layer, trainVar['w3']) + trainVar['b3']
    layer3_actv = tf.sigmoid (layer3_fc)

    layer4_fc = tf.matmul (layer3_actv, trainVar['w4']) + trainVar['b4']
    layer4_actv = tf.sigmoid (layer4_fc)

    logits = tf.matmul (layer4_actv, trainVar['w5']) + trainVar['b5']

    target = tf.placeholder(tf.float32, shape=(nbatch,10))

    loss = tf.losses.softmax_cross_entropy (
        target, 
        tf.cast(logits, tf.float32)
    )

    return data, target, loss
    
def main() : 
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # (55000x784)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images 
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    numSteps = 10000
    displayStep = 500      
    learningRate = 0.001
    nBatch = 128
    lossScale = 128
    dtype = tf.float16
    
    with tf.variable_scope (
        'fp32_storage', 
        custom_getter= float32_variable_storage_getter
    ) : 
        trainVar = variables_lenet5 (dtype)
        data, target, loss = model_lenet5 (dtype, trainVar, nBatch)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grads = gradients_with_loss_scaling (loss, variables, lossScale)
        optimiser = tf.train.GradientDescentOptimizer (learningRate)
        trainingStep = optimiser.apply_gradients(zip(grads, variables))
        
    sess = tf.Session () 
    sess.run (tf.global_variables_initializer())
   
    for step in xrange(numSteps) : 
        offset = step * nBatch
        batch_data = np.reshape(train_data[offset:(offset+nBatch),:], (-1,28,28,1))
        batch_labels = np.eye(10)[np.array(train_labels[offset:(offset+nBatch)]).reshape(-1)]
        feed_dict = {data: batch_data, target: batch_labels}
        step_loss, _ = sess.run([loss, trainingStep], feed_dict=feed_dict)
        print ('%4i %6f' % (step + 1, step_loss))

if __name__ == "__main__" : 
    main()
