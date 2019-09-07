#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time


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
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    pool3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    pool4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    conv7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
   
    return input_image, keep_prob, pool3, pool4, conv7
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
    # Prepare conv7 for the fusion
    # 1x1 conv to reduce the number of filters from 4096 to the number of classes for our specific model
    conv7_reduced = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', 
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # TODO: Init Upconv to bilinear interpolation
    # Increase spatial dimensions of the layer to match with the layer further back in the network for fusion
    conv7x2 = tf.layers.conv2d_transpose(conv7_reduced, num_classes, 4, strides=(2, 2), padding='same', 
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Prepare pool4 for the fusion
    # First scaling it (gives higher IoU and accuracy, according to the notes from pierluigi.ferrari)
    pool4_scaled = tf.multiply(vgg_layer4_out, 0.01, name='pool4_scaled')
    # Then reduce the number of filters to the number of classes
    pool4_reduced = tf.layers.conv2d(pool4_scaled, num_classes, 1, padding='same', 
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    
    # Fuse the layers (Skip Connection)
    pool4_conv7x2 = tf.add(pool4_reduced, conv7x2)
    
    # Upscale pool4_conv7x2
    pool4x2_conv7x4 = tf.layers.conv2d_transpose(pool4_conv7x2, num_classes, 4, strides=(2, 2), padding='same', 
                                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Prepare pool3 for the fusion
    # Scale
    pool3_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_scaled')
    # Layers reduction
    pool3_reduced = tf.layers.conv2d(pool3_scaled, num_classes, 1, padding='same', 
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Final fusion
    pool3_pool4x2_conv7x4 = tf.add(pool3_reduced, pool4x2_conv7x4)
    
    # Final upsample
    out = tf.layers.conv2d_transpose(pool3_pool4x2_conv7x4, num_classes, 16, strides=(8, 8), padding='same', 
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return out
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
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_last_layer, 
                                                                                   labels=correct_label))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = cross_entropy_loss + sum(reg_losses)
    
    # Adam optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    
    return nn_last_layer, train_op, loss
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
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        print("Epoch:", epoch+1)
        t0 = time.time()
        
        loss_sum = 0
        minibatch_count = 0
        
        for images, labels in get_batches_fn(batch_size):
            
            feed_dict={input_image: images, correct_label: labels, keep_prob: 0.5, learning_rate: 0.0001}
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict)
            
            loss_sum += loss
            minibatch_count += 1
            
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Train Loss =", loss_sum/minibatch_count)

    print("Done training!")

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        # Placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        # Load the VGG16
        input_image, keep_prob, pool3, pool4, conv7 = load_vgg(sess, vgg_path)
        # Add new layers
        layer_output = layers(pool3, pool4, conv7, num_classes)
        # Create optimizer
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        epochs = 10
        batch_size = 17
        
        saver = tf.train.Saver()
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
            
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
