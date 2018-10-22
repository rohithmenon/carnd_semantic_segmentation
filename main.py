#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


KEEP_PROB = 0.5
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 16
NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg_graph = tf.get_default_graph()

    layer_in = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3_out = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4_out = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7_out = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return layer_in, keep_prob, layer_3_out, layer_4_out, layer_7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # fcn8
    fcn8 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, name='fcn8')
    # fcn16 (layer4 skip with 1x1 convolution + fcn8 upsampled)
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, name='layer4_1x1')
    fcn8_upsampled = tf.layers.conv2d_transpose(fcn8, filters=num_classes, kernel_size=4, strides=(2, 2), padding='SAME', name='fcn8-upsampled') 
    fcn16 = tf.add(fcn8_upsampled, layer4_1x1, name='fcn16')
    # fcn32 (layer3 skip with 1x1 convolution + fc16 upsampled)
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, name='layer3_1x1')
    fcn16_upsampled = tf.layers.conv2d_transpose(fcn16, filters=num_classes, kernel_size=4, strides=(2, 2), padding='SAME', name='fcn16-upsampled') 
    fcn32 = tf.add(fcn16_upsampled, layer3_1x1, name='fcn32')
    fcn32_upsampled = tf.layers.conv2d_transpose(fcn32, filters=num_classes, kernel_size=16, strides=(8, 8), padding='SAME', name='fcn32-upsampled') 
    return fcn32_upsampled
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_last_layer, labels=correct_label)
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="train_op")

    return nn_last_layer, train_op, loss_op
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        total_loss = 0.0
        for img_batch, lbl_batch in get_batches_fn(batch_size):
            loss, grads = sess.run([cross_entropy_loss, train_op],
                                   feed_dict={input_image: img_batch,
                                              correct_label: lbl_batch,
                                              keep_prob: KEEP_PROB,
                                              learning_rate: LEARNING_RATE})
            total_loss += loss
            print("Epoch: {}, Loss: {}, TotalLoss: {}".format(epoch, loss, total_loss))

tests.test_train_nn(train_nn)


def run():
    data_dir = './'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        logits = layers(layer3, layer4, layer7, NUM_CLASSES)
        logits, train_op, loss_op = optimize(logits, correct_label, learning_rate, NUM_CLASSES) 

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Train NN using the train_nn function
        train_nn(sess,
                 EPOCHS,
                 BATCH_SIZE,
                 get_batches_fn,
                 train_op,
                 loss_op,
                 input_image,
                 correct_label,
                 keep_prob,
                 learning_rate)

        # Save inference data
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
