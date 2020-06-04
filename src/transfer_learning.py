#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 16, 2017
# Description  : trasnfer learning
#==============================================================================

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

# import os
# import numpy as np
# from PIL import Image

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

from src.nets.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from src.nets.inception_v3 import inception_v3, inception_v3_arg_scope
from src.nets.vgg import vgg_19, vgg_arg_scope
from src.utils import logging

slim = tf.contrib.slim

inceptionV3_layers = ['InceptionV3/Logits', 'InceptionV3/AuxLogits', 'InceptionV3/Mixed_7c', 'InceptionV3/Mixed_7b',
               'InceptionV3/Mixed_7a', 'InceptionV3/Mixed_6e', 'InceptionV3/Mixed_6d', 'InceptionV3/Mixed_6c',
               'InceptionV3/Mixed_6b', 'InceptionV3/Mixed_6a', 'InceptionV3/Mixed_5d', 'InceptionV3/Mixed_5c',
               'InceptionV3/Mixed_5b', 'InceptionV9/MaxPool_5a_3x3', 'InceptionV8/Conv2d_4a_3x3', 'InceptionV7/Conv2d_3b_1x1',
               'InceptionV3/MaxPool_3a_3x3', 'InceptionV3/Conv2d_2b_3x3', 'InceptionV3/Conv2d_2a_3x3', 'InceptionV3/Conv2d_1a_3x3']

inceptionResnetV2_layers = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Mixed_7a', 'InceptionResnetV2/Mixed_6a',
                     'InceptionResnetV2/Mixed_5b', 'InceptionResnetV2/MaxPool_5a_3x3', 'InceptionResnetV2/Conv2d_4a_3x3',
                     'InceptionResnetV2/Conv2d_3b_1x1', 'InceptionResnetV2/MaxPool_3a_3x3', 'InceptionResnetV2/Conv2d_2b_3x3',
                     'InceptionResnetV2/Conv2d_2a_3x3', 'InceptionResnetV2/Conv2d_1a_3x3']

vgg_19_layers = ['vgg_19/fc8', 'vgg_19/fc7', 'vgg_19/fc6', 'vgg_19/conv5', 'vgg_19/conv4', 'vgg_19/conv3', 'vgg_19/conv2', 'vgg_19/conv1']

train_accuracy = {}
test_accuracy = {}

optimizer_functions = {'gd': tf.train.GradientDescentOptimizer, 'adadelta': tf.train.AdadeltaOptimizer,
                       'adagrad': tf.train.AdagradOptimizer, 'adam': tf.train.AdamOptimizer, 'rmsprop': tf.train.RMSPropOptimizer}

# transfer learning - training
def train(mode, dataset, images, labels, batch_size, num_epochs, optimizer_fn, learning_rate, learning_rate_decay_factor, num_epochs_per_decay,
          dropout_keep_prob, pretrained_model, model_dir, pretrained_model_dir, layer_count, logger = None):
    try:
        # find the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size) + 1
        num_steps_per_epoch = num_batches_per_epoch  # one step is one batch processed
        # decay_steps = int(num_epochs_per_decay * num_steps_per_epoch)

        # initializing the model
        if (pretrained_model == 'inceptionV3'):  # inception V3
            model_file = 'inception_v3.ckpt'
            architecture_layers = inceptionV3_layers
            with slim.arg_scope(inception_v3_arg_scope()):  # create the model inference
                logits, end_points = inception_v3(images, num_classes=dataset.num_classes, dropout_keep_prob=dropout_keep_prob, is_training=(mode == 'train'))
        elif (pretrained_model == 'inception_resnetV2'):  # inception_resnetV2
            model_file = 'inception_resnet_v2_2016_08_30.ckpt'
            architecture_layers = inceptionResnetV2_layers
            with slim.arg_scope(inception_resnet_v2_arg_scope()):  # create the model inference
                logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, dropout_keep_prob=dropout_keep_prob, is_training=(mode == 'train'))
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            model_file = 'vgg_19.ckpt'
            architecture_layers = vgg_19_layers
            with slim.arg_scope(vgg_arg_scope()):  # create the model inference
                logits, end_points = vgg_19(images, num_classes=dataset.num_classes, dropout_keep_prob=dropout_keep_prob, is_training=(mode == 'train'))

        if (pretrained_model == 'inceptionV3' or pretrained_model == 'inception_resnetV2'):  # inceptionV3 or inception_resnetV2
            logging("Transfer learning layers-" + str(layer_count) + ": " + str(architecture_layers[:(layer_count + 1)]), logger, 'info')

            # define the scopes that you want to exclude for restoration
            variables_to_restore = slim.get_variables_to_restore(exclude=architecture_layers[:(layer_count + 1)])
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            logging("Transfer learning layers-" + str(layer_count) + ": " + str(architecture_layers[:(layer_count)]), logger, 'info')

            # define the scopes that you want to exclude for restoration
            variables_to_restore = slim.get_variables_to_restore(exclude=architecture_layers[:(layer_count)])

        # perform one-hot-encoding of the labels
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        # performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    # obtain the regularization losses as well

        # global_step = get_or_create_global_step()           # create the global step for monitoring the learning_rate and training.

        # define your exponentially decaying learning rate
        # learning_rate = tf.train.exponential_decay(
        #     learning_rate = initial_learning_rate,
        #     global_step = global_step,
        #     decay_steps = decay_steps,
        #     decay_rate = learning_rate_decay_factor,
        #     staircase = True)

        optimizer = optimizer_functions[optimizer_fn](learning_rate = learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)              # define the optimizer that takes on the learning rate
        train_op = slim.learning.create_train_op(total_loss, optimizer)     # create the train_op

        # the metrics that need to predict it isn't one_hot_encoded.
        if (pretrained_model == 'inceptionV3' or pretrained_model == 'inception_resnetV2'):  # inceptionV3 or inception_resnetV2
            predictions = tf.argmax(end_points['Predictions'], 1)
            # probabilities = end_points['Predictions']
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            predictions = tf.cast(tf.to_int64(tf.argmax(logits, 1)), tf.float32)

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update) #, probabilities)

        # all the summaries need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', loss)
        tf.summary.scalar('train_accuracy', accuracy)
        # tf.summary.scalar('learning_rate', learning_rate)
        summary_op = tf.summary.merge_all()

        # create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=1)

        def restore_fn(sess):
            return saver.restore(sess, pretrained_model_dir + model_file)

        # define supervisor for running a managed session.
        sv = tf.train.Supervisor(logdir=model_dir + str(layer_count) + "/", summary_op=None, init_fn=restore_fn)

        # run the managed session
        with sv.managed_session() as sess:
            # sess.run(init_from_final_layer)  # initialize the last unfreez layer

            max_train_accuracy = 0.0
            for step in range(num_steps_per_epoch * num_epochs):
                loss_value, _, _ = sess.run([train_op, sv.global_step, metrics_op])

                if(step == 0):
                    logging('Step: ' + str(int(step / num_batches_per_epoch + 1)) + '/' + str(num_epochs) + ' | learning rate: ' + str(learning_rate), logger, 'info')

                if step % num_steps_per_epoch == 0 and step != 0:
                    logits_value, loss_value, accuracy_value, summary_values = sess.run([logits, total_loss, accuracy, summary_op])

                    logging('Step: ' + str(int(step / num_batches_per_epoch + 1)) + '/' + str(num_epochs) + ' | loss: ' + str(loss_value) +
                            ' | accuracy: ' + str(accuracy_value), logger, 'info')      #  + ' | learning rate: ' + str(learning_rate_value)

                    sv.summary_computed(sess, summary_values)       # log the summaries

                    if(accuracy_value > max_train_accuracy):
                        max_train_accuracy = accuracy_value

            # log the final training loss and accuracy
            total_loss_value, total_accuracy_value = sess.run([total_loss, accuracy])
            logging('Training Final loss: ' + str(total_loss_value) + ' | Training Final accuracy: ' + str(max_train_accuracy), logger, 'info')

            # once all the training has been done, save the log files and checkpoint model
            logging('Saving model of layers-' + str(layer_count), logger, 'info')
            sv.saver.save(sess, model_dir + str(layer_count) + '/', global_step=sv.global_step)

            if not str(layer_count) in train_accuracy:
                train_accuracy[str(layer_count)] = str(num_epochs) + '\t' + str(round(learning_rate, 8)) + '\t' + str(total_loss_value) + \
                                                   '\t' + str(total_accuracy_value * 100) + '\t' + str(max_train_accuracy * 100)

        logging("Transfer learning training completed successfully", logger, 'info')

        return train_accuracy
    except Exception as e:
        logging("Trasnfer learning training failed - " + str(e), logger, 'error')

def test(mode, dataset, images, labels, batch_size, num_epochs, pretrained_model, model_dir, layer_count, logger = None):
    try:
        # inception and inception_resnet only
        num_batches_per_epoch = int(dataset.num_samples / batch_size) + 1
        num_steps_per_epoch = num_batches_per_epoch

        # initializing the model
        if (pretrained_model == 'inceptionV3'):  # inception V3
            architecture_layers = inceptionV3_layers
            with slim.arg_scope(inception_v3_arg_scope()):  # create the model inference
                logits, end_points = inception_v3(images, num_classes=dataset.num_classes, is_training=False)
        elif (pretrained_model == 'inception_resnetV2'):  # inception_resnetV2
            architecture_layers = inceptionResnetV2_layers
            with slim.arg_scope(inception_resnet_v2_arg_scope()):  # create the model inference
                logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=False)
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            architecture_layers = vgg_19_layers
            with slim.arg_scope(vgg_arg_scope()):  # create the model inference
                logits, end_points = vgg_19(images, num_classes=dataset.num_classes, is_training=True) #=(mode == 'train'))

        if (pretrained_model == 'inceptionV3' or pretrained_model == 'inception_resnetV2'):  # inceptionV3 or inception_resnetV2
            logging("Transfer learning testing layers-" + str(layer_count) + ": " + str(architecture_layers[:(layer_count + 1)]), logger, 'info')
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            logging("Transfer learning testing layers-" + str(layer_count) + ": " + str(architecture_layers[:(layer_count)]), logger, 'info')

        # the metrics that need to predict it isn't one_hot_encoded.
        if (pretrained_model == 'inceptionV3' or pretrained_model == 'inception_resnetV2'):  # inceptionV3 or inception_resnetV2
            predictions = tf.argmax(end_points['Predictions'], 1) #end_points['Predictions'], 1)
            # probabilities = end_points['Predictions']
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            predictions = tf.cast(tf.to_int64(tf.argmax(logits, 1)), tf.float32)

        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update) #, probabilities)

        # create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)  # no apply_gradient method so manually increasing the global_step

        # define some scalar quantities to monitor
        tf.summary.scalar('test_accuracy', accuracy)
        # summary_op = tf.summary.merge_all()

        # get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)

        def restore_fn(sess):
            saver.restore(sess, tf.train.latest_checkpoint(model_dir + str(layer_count) + '/'))

        # get the supervisor
        sv = tf.train.Supervisor(logdir=model_dir + str(layer_count) + '/', summary_op=None, saver=None, init_fn=restore_fn)

        # run in one session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)

                if step == 0:
                    sess.run(accuracy)
                else:
                    _, global_step_count, accuracy_value = sess.run([metrics_op, sv.global_step, accuracy])

                if step != 0 and step % 10 == 0:
                    logging('Step: ' + str(step) + ' | test accuracy: ' + str(accuracy_value), logger, 'info')

                    # sv.summary_computed(sess, sess.run(summary_op))


            # at the end of all the evaluation, show the final accuracy
            total_accuracy = sess.run(accuracy)
            logging('Testing Final accuracy: ' + str(total_accuracy), logger, 'info')

            if not str(layer_count) in test_accuracy:
                test_accuracy[str(layer_count)] = str(total_accuracy * 100)

            # visualize the last batch's images just to see what our model has predicted
            # raw_image, labels, predictions = sess.run([raw_image, labels, predictions])
            # for i in range(10):
            #     image, label, prediction = raw_image[i], labels[i], predictions[i]
            #     prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
            #     text = 'Prediction: %s \n Ground Truth: %s' % (prediction_name, label_name)
            #     # print(text)
            #     img_plot = plt.imshow(image)
            #
            #     # set up the plot and hide axes
            #     plt.title(text)
            #     img_plot.axes.get_yaxis().set_ticks([])
            #     img_plot.axes.get_xaxis().set_ticks([])
            #     plt.show()

        logging("Transfer learning testing completed successfully, accuracy: " + str(total_accuracy * 100), logger, 'info')

        return test_accuracy
    except Exception as e:
        logging("Transfer learning testing failed - " + str(e), logger, 'error')
        test_accuracy[str(layer_count)] = 0.0
        return test_accuracy