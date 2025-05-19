# Image Transformation Network - Deep Residual Convolutional Neural Network
# Used for Style Transferring

from __future__ import division
import tensorflow as tf
try:
    tf1 = tf.compat.v1
    tf1.disable_v2_behavior()
except AttributeError:
    tf1 = tf  # For TF1.x fallback
from model import VGG, preprocess
import numpy as np
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from imageio import imread, imwrite
from PIL import Image

WEIGHT_INIT_STDDEV = 0.1


def conv2d(x, input_filters, output_filters, kernel_size, strides, relu=True, mode='REFLECT'):
    shape  = [kernel_size, kernel_size, input_filters, output_filters]
    weight = tf.Variable(tf1.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

    padding  = kernel_size // 2
    x_padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]], mode=mode)

    out = tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID')

    out = instance_norm(out, output_filters)

    if relu:
        out = tf.nn.relu(out)

    return out


def conv2d_transpose(x, input_filters, output_filters, kernel_size, strides):
    shape  = [kernel_size, kernel_size, output_filters, input_filters]
    weight = tf.Variable(tf1.truncated_normal(shape, stddev=WEIGHT_INIT_STDDEV), name='weight')

    batch_size = tf.shape(x)[0]
    height     = tf.shape(x)[1] * strides
    width      = tf.shape(x)[2] * strides

    output_shape = [batch_size, height, width, output_filters]

    out = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1])

    out = instance_norm(out, output_filters)

    out = tf.nn.relu(out)

    return out


def instance_norm(x, num_filters):
    epsilon = 1e-3

    shape = [num_filters]
    scale = tf.Variable(tf.ones(shape), name='scale')
    shift = tf.Variable(tf.zeros(shape), name='shift')

    # Use keepdims instead of keep_dims for TF >= 1.0
    mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
    x_normed  = tf1.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    return scale * x_normed + shift


def residual(x, filters, kernel_size, strides):
    conv1 = conv2d(x, filters, filters, kernel_size, strides)
    conv2 = conv2d(conv1, filters, filters, kernel_size, strides, relu=False)

    return x + conv2


def transform(image):
    image = image / 127.5 - 1

    # mitigate border effects via padding a little before passing through
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    # Use tf1.variable_scope for compatibility
    with tf1.variable_scope('conv1'):
        conv1 = conv2d(image, 3, 32, 9, 1)
    with tf1.variable_scope('conv2'):
        conv2 = conv2d(conv1, 32, 64, 3, 2) # with downsampling
    with tf1.variable_scope('conv3'):
        conv3 = conv2d(conv2, 64, 128, 3, 2) # with downsampling

    with tf1.variable_scope('residual1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf1.variable_scope('residual2'):
        res2 = residual(res1, 128, 3, 1)
    with tf1.variable_scope('residual3'):
        res3 = residual(res2, 128, 3, 1)
    with tf1.variable_scope('residual4'):
        res4 = residual(res3, 128, 3, 1)
    with tf1.variable_scope('residual5'):
        res5 = residual(res4, 128, 3, 1)

    with tf1.variable_scope('deconv1'):
        deconv1 = conv2d_transpose(res5, 128, 64, 3, 2) # with upsampling
    with tf1.variable_scope('deconv2'):
        deconv2 = conv2d_transpose(deconv1, 64, 32, 3, 2) # with upsampling
    with tf1.variable_scope('convout'):
        convout = tf.tanh(conv2d(deconv2, 32, 3, 9, 1, relu=False))

    output = (convout + 1) * 127.5

    # remove border effects via reducing padding
    height = tf.shape(output)[1]
    width  = tf.shape(output)[2]

    output = tf.slice(output, [0, 10, 10, 0], [-1, height - 20, width - 20, -1])

    return output


# Utility


def imresize(arr, size, interp=None):
    # Ignore interp argument, always use BILINEAR for compatibility
    return np.array(Image.fromarray(arr).resize(size, Image.BILINEAR))


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = imread(path, mode='RGB')

        if height is not None and width is not None:
            # Remove interp argument, just pass size
            image = imresize(image, [height, width])

        images.append(image)

    images = np.stack(images, axis=0)

    return images


def save_images(paths, datas, save_path, prefix=None, suffix=None):
    if isinstance(paths, str):
        paths = [paths]

    assert len(paths) == len(datas)

    if not exists(save_path):
        mkdir(save_path)

    if prefix is None:
        prefix = ''
    if suffix is None:
        suffix = ''

    for i, path in enumerate(paths):
        data = datas[i]

        # Fix shape and dtype
        data = np.squeeze(data)  # remove singleton dimensions
        data = np.clip(data, 0, 255)  # clip to [0, 255]
        data = data.astype(np.uint8)  # convert to uint8

        name, ext = splitext(path)
        name = name.split(sep)[-1]
        ext = '.jpg'  # Save as .jpg to avoid unsupported formats

        save_file_path = join(save_path, prefix + name + suffix + ext)

        imwrite(save_file_path, data)


# Train the Image Transform Net using a fixed VGG19 as a Loss Network
# The VGG19 is pre-trained on ImageNet dataset

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-3


def train(content_targets_path, style_target_path, content_weight, style_weight, tv_weight, vgg_path, save_path, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # guarantee the size of content_targets is a multiple of BATCH_SIZE
    mod = len(content_targets_path) % BATCH_SIZE
    if mod > 0:
        print('Train set has been trimmed %d samples...' % mod)
        content_targets_path = content_targets_path[:-mod]

    height, width, channels = TRAINING_IMAGE_SHAPE
    input_shape = (BATCH_SIZE, height, width, channels)

    # create a pre-trained VGG network
    vgg = VGG(vgg_path)

    # retrive the style_target image
    style_target = get_images(style_target_path) # shape: (1, height, width, channels)
    style_shape  = style_target.shape

    # compute the style features
    style_features = {}
    with tf.Graph().as_default(), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')

        # pass style_image through 'pretrained VGG-19 network'
        style_img_preprocess = preprocess(style_image)
        style_net = vgg.forward(style_img_preprocess)

        for style_layer in STYLE_LAYERS:
            features = style_net[style_layer].eval(feed_dict={style_image: style_target})
            features = np.reshape(features, [-1, features.shape[3]])

            gram = np.matmul(features.T, features) / features.size
            style_features[style_layer] = gram

    # compute the perceptual losses
    with tf.Graph().as_default(), tf.Session() as sess:
        content_images = tf.placeholder(tf.float32, shape=input_shape, name='content_images')

        # pass content_images through 'pretrained VGG-19 network'
        content_imgs_preprocess = preprocess(content_images)
        content_net = vgg.forward(content_imgs_preprocess)

        # compute the content features
        content_features = {}
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        # pass content_images through 'Image Transform Net'
        output_images = transform(content_images)

        # pass output_images through 'pretrained VGG-19 network'
        output_imgs_preprocess = preprocess(output_images)
        output_net = vgg.forward(output_imgs_preprocess)

        # ** compute the feature reconstruction loss **
        content_size = tf.size(content_features[CONTENT_LAYER])

        content_loss = 2 * tf.nn.l2_loss(output_net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / tf.to_float(content_size)

        # ** compute the style reconstruction loss **
        style_losses = []
        for style_layer in STYLE_LAYERS:
            features = output_net[style_layer]
            shape = tf.shape(features)
            num_images, height, width, num_filters = shape[0], shape[1], shape[2], shape[3]
            features = tf.reshape(features, [num_images, height*width, num_filters])
            grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(height * width * num_filters)
            style_gram = style_features[style_layer]
            layer_style_loss = 2 * tf.nn.l2_loss(grams - style_gram) / tf.to_float(tf.size(grams))
            style_losses.append(layer_style_loss)

        style_loss = tf.reduce_sum(tf.stack(style_losses))

        # ** compute the total variation loss **
        shape = tf.shape(output_images)
        height, width = shape[1], shape[2]
        y = tf.slice(output_images, [0, 0, 0, 0], [-1, height - 1, -1, -1]) - tf.slice(output_images, [0, 1, 0, 0], [-1, -1, -1, -1])
        x = tf.slice(output_images, [0, 0, 0, 0], [-1, -1,  width - 1, -1]) - tf.slice(output_images, [0, 0, 1, 0], [-1, -1, -1, -1])

        tv_loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

        # overall perceptual losses
        loss = content_weight * content_loss + style_weight * style_loss + tv_weight * tv_loss

        # Training step
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        n_batches = len(content_targets_path) // BATCH_SIZE

        if debug:
            elapsed_time = datetime.now() - start_time
            tf.logging.set_verbosity(tf.logging.INFO)
            tf.logging.info('Elapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            tf.logging.info('Now begin to train the model...')
            start_time = datetime.now()

        for epoch in range(EPOCHS):

            np.random.shuffle(content_targets_path)

            for batch in range(n_batches):
                # retrive a batch of content_targets images
                content_batch_path = content_targets_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]
                content_batch = get_images(content_batch_path, input_shape[1], input_shape[2])

                # run the training step
                sess.run(train_op, feed_dict={content_images: content_batch})

                step += 1

                if step % 1000 == 0:
                    saver.save(sess, save_path, global_step=step)

                if debug:
                    is_last_step = (epoch == EPOCHS - 1) and (batch == n_batches - 1)

                    if is_last_step or step % logging_period == 0:
                        elapsed_time = datetime.now() - start_time
                        _content_loss, _style_loss, _tv_loss, _loss = sess.run([content_loss, style_loss, tv_loss, loss], feed_dict={content_images: content_batch})

                        tf.logging.info('step: %d,  total loss: %f,  elapsed time: %s' % (step, _loss, elapsed_time))
                        tf.logging.info('content loss: %f,  weighted content loss: %f' % (_content_loss, content_weight * _content_loss))
                        tf.logging.info('style loss  : %f,  weighted style loss  : %f' % (_style_loss, style_weight * _style_loss))
                        tf.logging.info('tv loss     : %f,  weighted tv loss     : %f' % (_tv_loss, tv_weight * _tv_loss))
                        tf.logging.info('\n')

        # ** Done Training & Save the model **
        saver.save(sess, save_path)

        if debug:
            elapsed_time = datetime.now() - start_time
            tf.logging.info('Done training! Elapsed time: %s' % elapsed_time)
            tf.logging.info('Model is saved to: %s' % save_path)


# Use a trained Image Transform Net to generate
# a style transferred image with a specific style




# from utils import get_images, save_images


def generate(contents_path, model_path, is_same_size=False, resize_height=None, resize_width=None, save_path=None, prefix='stylized-', suffix=None):
    if isinstance(contents_path, str):
        contents_path = [contents_path]

    if is_same_size or (resize_height is not None and resize_width is not None):
        outputs = _handler1(contents_path, model_path, resize_height=resize_height, resize_width=resize_width, save_path=save_path, prefix=prefix, suffix=suffix)
        return list(outputs)
    else:
        outputs = _handler2(contents_path, model_path, save_path=save_path, prefix=prefix, suffix=suffix)
        return outputs


def _handler1(content_path, model_path, resize_height=None, resize_width=None, save_path=None, prefix=None, suffix=None):
    # get the actual image data, output shape: (num_images, height, width, color_channels)
    content_target = get_images(content_path, resize_height, resize_width)

    with tf1.Graph().as_default(), tf1.Session() as sess:
        # build the dataflow graph
        content_image = tf1.placeholder(tf.float32, shape=content_target.shape, name='content_image')

        output_image = transform(content_image)

        # restore the trained model and run the style transferring
        saver = tf1.train.Saver()
        saver.restore(sess, model_path)

        output = sess.run(output_image, feed_dict={content_image: content_target})

    if save_path is not None:
        save_images(content_path, output, save_path, prefix=prefix, suffix=suffix)

    return output


def _handler2(content_path, model_path, save_path=None, prefix=None, suffix=None):
    with tf1.Graph().as_default(), tf1.Session() as sess:
        # build the dataflow graph
        content_image = tf1.placeholder(tf.float32, shape=(1, None, None, 3), name='content_image')

        output_image = transform(content_image)

        # restore the trained model and run the style transferring
        saver = tf1.train.Saver()
        saver.restore(sess, model_path)

        output = []
        for content in content_path:
            content_target = get_images(content)
            result = sess.run(output_image, feed_dict={content_image: content_target})
            output.append(result[0])

    if save_path is not None:
        save_images(content_path, output, save_path, prefix=prefix, suffix=suffix)

    return output

# All code is ready for import and use by main.py
model_dir = "/persistent_storage/models/model1"