# ==============================================================================
# Utility functions
# ==============================================================================

import os
# import numpy as np
from sklearn.metrics import confusion_matrix
# # from keras.layers import Activation, Conv2D
# # from keras.layers.normalization import BatchNormalization
# # from keras import backend as K
from enum import Enum
import logging as _logging
from tensorflow.python.client import device_lib
from configparser import ConfigParser
import tensorflow as tf
from tensorflow.python.training.device_setter import _RoundRobinStrategy
from tensorflow.python.framework import device as pydev
from tensorflow.core.framework import node_def_pb2
import operator

# Load configuration
def getConfig(config_file):
    try:
        parser = ConfigParser()
        parser.read(config_file)

        # get the ints, floats and strings
        _conf_ints = [(key, int(value)) for (key, value) in parser.items('ints')]
        _conf_floats = [(key, float(value)) for (key, value) in parser.items('floats')]
        _conf_strings = [(key, str(value)) for (key, value) in parser.items('strings')]
        return dict(_conf_ints + _conf_floats + _conf_strings)
    except Exception as e:
        print("Failed to load configuration - " + str(e))

# check a path exists or not
def checkPathExists(path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

# devices enum
class DeviceCategory(Enum):
    CPU = 1
    GPU = 2

# Used with tf.device() to place variables on the least loaded GPU
class GpuParamServerDeviceSetter(object):
    """
      A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
      'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
      placed on the least loaded gpu. All other Ops, which will be the computation
      Ops, will be placed on the worker_device.
    """

    def __init__(self, worker_device, ps_devices):
        """Initializer for GpuParamServerDeviceSetter.
        Args:
          worker_device: the device to use for computation Ops.
          ps_devices: a list of devices to use for Variable Ops. Each variable is
          assigned to the least loaded device.
        """
        self.ps_devices = ps_devices
        self.worker_device = worker_device
        self.ps_sizes = [0] * len(self.ps_devices)

    def __call__(self, op):
        if op.device:
            return op.device
        if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
            return self.worker_device

        # Gets the least loaded ps_device
        device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
        device_name = self.ps_devices[device_index]
        var_size = op.outputs[0].get_shape().num_elements()
        self.ps_sizes[device_index] += var_size

        return device_name


# get gpu devices
def get_gpu_devices():
    local_device_protos = device_lib.list_local_devices()
    gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpu_devices


# get cpu devices
def get_cpu_devices():
    local_device_protos = device_lib.list_local_devices()
    cpu_devices = [x.name for x in local_device_protos if x.device_type == 'CPU']
    return cpu_devices


# Create device setter object
def create_device_setter(device_category: DeviceCategory, device: str, gpu_devices: list):
    if device_category == DeviceCategory.CPU:
        # tf.train.replica_device_setter supports placing variables on the CPU, all on one GPU, or on ps_servers defined in a cluster_spec.
        return tf.train.replica_device_setter(worker_device=device, ps_device='/cpu:0', ps_tasks=1)
    else:
        return GpuParamServerDeviceSetter(device, gpu_devices)


# device setter
def get_device_setter(device_category: DeviceCategory, device):
    if device_category == DeviceCategory.GPU:
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(get_gpu_devices()[0]),
                                                                      tf.contrib.training.byte_size_load_fn)
    else:
        ps_strategy = _RoundRobinStrategy(len(get_cpu_devices()[0]))

    ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string('/{}:{}'.format(device_category.name, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser

# logging
def logging(message, logging, type='error'):
    try:
        if (type == 'debug'):
            _logging.debug(message)
        elif (type == 'info'):
            _logging.info(message)
        elif (type == 'warning'):
            _logging.warning(message)
        elif (type == 'error'):
            _logging.error(message)
        elif (type == 'critical'):
            _logging.critical(message)

        print(message)
    except Exception as e:
        print("Logging failed - " + str(e))


# check a path exists or not
def checkPathExists(path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)
            
# generate the One-Hot encoded class-labels from an array of integers
# def oneHotEncoded(class_numbers, num_classes = None, logger = None):
#     try:
#         if(num_classes == None):
#             num_classes = np.max(class_numbers) + 1
#
#         return np.eye(num_classes, dtype=float)[class_numbers]
#     except Exception as e:
#         logging("One-hot-encoding failed - " + str(e), logger, 'error')

# Convert images from the CIFAR-10 format and return a 4-dim array with shape: [image_number, height, width, channel]
# where the pixels are floats between 0.0 and 1.0
# def reshapeImage(images, num_channels, img_size, logger = None):
#     try:
#         raw_float = np.array(images, dtype = float) / 255.0  # convert the raw images from the data-files to floating-points
#         images = raw_float.reshape([-1, num_channels, img_size, img_size])  # reshape the array to 4-dimensions
#         image_batch = images.transpose([0, 2, 3, 1])
#
#         return image_batch
#     except Exception as e:
#         logging("Reshape image - " + str(e), logger, 'error')
        
# function to apply conv + BN
# def conv2d_bn(x, filters, num_row, num_col, padding = 'same', strides = (1, 1), name = None, logger = None):
#     try:
#         if name is not None:
#             bn_name = name + '_bn'
#             conv_name = name + '_conv'
#         else:
#             bn_name = None
#             conv_name = None
#         if K.image_data_format() == 'channels_first':
#             bn_axis = 1
#         else:
#             bn_axis = 3
#         x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
#         x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
#         x = Activation('relu', name=name)(x)
#         return x
#     except Exception as e:
#         logging("Conv 2D bn failed - " + str(e), logger, 'error')
#
# # function for selecting a random batch of transfer-values from the training-set
# def randomBatch(train_batch_size, labels_train, transfer_values_train, logger = None):
#     try:
#         num_images = len(transfer_values_train)     # number of images (transfer-values) in the training-set
#
#         # Create a random index.
#         idx = np.random.choice(num_images, size = train_batch_size, replace = False)    # create a random index
#
#         x_batch = transfer_values_train[idx]    # use the random index to select random x and y-values. We use the transfer-values instead of images as x-values
#         y_batch = labels_train[idx]
#
#         return x_batch, y_batch
#     except Exception as e:
#         logging("Random batch failed - " + str(e), logger, 'error')
        
# plot confusion matrix
def plotConfusionMatrix(cls_pred, cls_test, num_classes, class_names, logger = None):
    try:
        # cls_pred is an array of the predicted class-number for all images in the test-set.
    
        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_test, y_pred=cls_pred)  # Predicted class.
    
        print("Confusion Matrix:")
        print("Confusion Matrix:")
        # Print the confusion matrix as text.
        for i in range(num_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)
            
    
        # Print the class-numbers for easy reference.
        class_numbers = [" ({0})".format(i) for i in range(num_classes)]
        print("".join(class_numbers))
        
        cm.stats()
        
        print(cm.print_stats())
    except Exception as e:
        logging("Failed to plot confusion matrix - " + str(e), logger, 'error')
        
# calculating classifications
# def predictClassification(sess, x, y_true, y_pred_cls, transfer_values, labels, cls_true, batch_size = 256, logger = None):
#     try:
#         # Number of images
#         num_images = len(transfer_values)
#
#         # Allocate an array for the predicted classes which will be calculated in batches and filled into this array.
#         cls_pred = np.zeros(shape=num_images, dtype=np.int)
#
#         # The starting index for the next batch is denoted i.
#         i = 0
#
#         while i < num_images:
#             # The ending index for the next batch is denoted j.
#             j = min(i + batch_size, num_images)
#
#             # Create a feed-dict with the images and labels between index i and j.
#             feed_dict = {x: transfer_values[i:j], y_true: labels[i:j]}
#
#             # Calculate the predicted class using TensorFlow.
#             cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)
#
#             # Set the start-index for the next batch to the end-index of the current batch.
#             i = j
#
#         correct = (cls_true == cls_pred)     # create a boolean array whether each image is correctly classified
#
#         return correct, cls_pred
#     except Exception as e:
#         logging("Predict classifications failed - " + str(e), logger, 'error')
        
# print test accuracy
# def testAccuracy(sess, x, y_true, y_pred_cls, transfer_values_test, labels_test, cls_test, images_test, num_classes,
#                  class_names, show_example_errors = False, show_confusion_matrix = False, logger = None):
#     try:
#         # For all the images in the test-set, calculate the predicted classes and whether they are correct.
#         correct, cls_pred = predictClassification(sess, x, y_true, y_pred_cls, transfer_values = transfer_values_test, labels = labels_test, cls_true = cls_test)
#
#         # Classification accuracy and the number of correct classifications.
#         acc = correct.mean()
#         num_correct = correct.sum()
#
#         # Number of images being classified.
#         num_images = len(correct)
#
#         # Print the accuracy.
#         msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
#         print(msg.format(acc, num_correct, num_images))
#
#         # Plot some examples of mis-classifications, if desired.
# #         if show_example_errors:
# #             print("Example errors:")
# #             plotExampleErrors(images_test, cls_test, cls_pred, correct, class_names)
#
#         # Plot the confusion matrix, if desired.
#         if show_confusion_matrix:
#             print("Confusion Matrix:")
#             plotConfusionMatrix(cls_pred, cls_test, num_classes, class_names)
#     except Exception as e:
#         logging("Test Accuracy failed - " + str(e), logger, 'error')

        
