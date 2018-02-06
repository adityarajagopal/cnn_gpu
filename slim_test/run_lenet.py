import lenet 
import time
from datasets import mnist 
from model import load_batch 
import tensorflow as tf
from tensorflow.python import debug as tf_debug

BATCHSIZE=5
LEARNING_RATE=0.001
NUM_STEPS=100
LOSS_SCALING_FACTOR=1

slim = tf.contrib.slim

def scale_down_grads(grads) : 
    return [(grad/LOSS_SCALING_FACTOR,var) for grad, var in grads]

def main(args) : 
    dataset = mnist.get_split('train', '/tmp/mnist')

    images, labels = load_batch (dataset, BATCHSIZE, is_training=True)

    with slim.arg_scope (lenet.lenet_arg_scope()) : 
        logits, end_points = lenet.lenet(images, is_training=True)

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

    tf.losses.softmax_cross_entropy(one_hot_labels, logits)

    total_loss = tf.losses.get_total_loss() * LOSS_SCALING_FACTOR

    tf.summary.scalar('loss', total_loss / LOSS_SCALING_FACTOR)
    
    optimiser = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    
    train_op = tf.contrib.training.create_train_op (
        total_loss, 
        optimiser, 
        summarize_gradients=True, 
        transform_grads_fn=scale_down_grads
    )

    for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) :
        print (i)

    slim.learning.train(
        train_op, 
        './log/train_3', 
        save_summaries_secs=2,
        #session_wrapper=tf_debug.LocalCLIDebugWrapperSession        
    )

if __name__ == '__main__' : 
    tf.app.run()

    
