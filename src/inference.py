#==============================================================================
# Author       : Abbas R. Ali
# Last modified: December 16, 2017
# Description  : inference
#==============================================================================

import numpy as np
from PIL import Image
import cv2
# from scipy import stats
from src.preprocessing import inception_preprocessing

import tensorflow as tf
from src.nets.inception_v3 import *
from src.nets.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from src.nets.vgg import vgg_19, vgg_arg_scope
# from preprocessing import vgg_preprocessing

from tensorflow.python.platform import gfile

slim = tf.contrib.slim

entropy = lambda p: -np.sum(p * np.log2(p))

#######################
# inception
#######################
# initialization
def InitializeInception(pretrained_model_dir):
    try:
        with tf.Session() as sess:
            with gfile.FastGFile(pretrained_model_dir + "/classify_image_graph_def.pb", "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name="")

        print("Inception network loaded successfully")
        return sess
    except Exception as e:
        print("Failed to initialize inception - " + str(e))

# classification
def ClassifyInception(sess, image_path):
    try:
        image_data = gfile.FastGFile(image_path, "rb").read()

        # tensors = sess.graph.get_tensor_by_name('pool_3:0') # 2048
        predictions = sess.graph.get_tensor_by_name('softmax:0')
        probabilities = sess.run([predictions], {"DecodeJpeg/contents:0": image_data})
        probabilities = np.squeeze(probabilities)

        ent = entropy(probabilities)
        entropy_ = lambda probabilities: -np.sum(probabilities * np.log2(probabilities))

        # ent1 = entropy_(probabilities)
        return max(probabilities), ent
    except Exception as e:
        print("Failed to classify image via inception - " + str(e))
        return 0.0, 0.0

# close session
def CloseInception(sess):
    try:
        sess.close()
    except Exception as e:
        print("Failed to close inception - " + str(e))

#######################
# Resnet and inception
#######################
# initialization
def InitializeInceptionResnet(pretrained_model_dir):
    try:
        image_size = inception_resnet_v2.default_image_size

        sess = tf.Session()
        input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(input_tensor, is_training = False)

        saver = tf.train.Saver()
        saver.restore(sess, pretrained_model_dir + 'inception_resnet_v2_2016_08_30.ckpt')

        with open(pretrained_model_dir + 'imagenet1000_clsid_to_human.txt','r') as inf:
            imagenet_classes = eval(inf.read())

        print("Inception_Resnet network loaded successfully")
        return(sess, end_points, logits, input_tensor, imagenet_classes)
    except Exception as e:
        print("Failed to initialize inception and resnet - " + str(e))

# classification
def ClassifyInceptionResnet(sess, end_points, logits, input_tensor, image_path):
    try:
        image_size = inception_resnet_v2.default_image_size

        processed_image = np.array(Image.open(image_path).resize((image_size, image_size)))
        processed_image = processed_image.reshape(-1, image_size, image_size, 3)
        processed_image = 2 * (processed_image / 255.0) - 1.0

        probabilities, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: processed_image})
        probabilities = np.squeeze(probabilities)

        ent = entropy(probabilities)

        return max(probabilities), ent
    except Exception as e:
        print("Failed to classify image via inception and resnet - " + str(e))
        return 0.0, 0.0

# close session
def CloseInceptionResnet(sess):
    try:
        sess.close()
    except Exception as e:
        print("Failed to close inception and resnet - " + str(e))

#######################
# VGG
#######################
# initialization
def InitializeVGG(pretrained_model_dir):
    try:
        image_size = vgg_19.default_image_size

        sess = tf.Session()
        input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        with slim.arg_scope(vgg_arg_scope()):
            logits, _ = vgg_19(input_tensor, num_classes = 1000, is_training = False)

        saver = tf.train.Saver()
        saver.restore(sess, pretrained_model_dir + 'vgg_19.ckpt')

        probabilities = tf.nn.softmax(logits)

        print("VGG network loaded successfully")
        return (sess, probabilities, input_tensor)
    except Exception as e:
        print("Failed to initialize vgg - " + str(e))

# classification
def ClassifyVGG(sess, prediction, input_tensor, image_path):
    try:
        image_size = vgg_19.default_image_size      # default VGG image size

        processed_image = np.array(Image.open(image_path).resize((image_size, image_size)))
        processed_image = processed_image.reshape(-1, image_size, image_size, 3)
        # processed_image = 2 * (processed_image / 255.0) - 1.0

        probabilities = sess.run([prediction], feed_dict={input_tensor: processed_image})
        probabilities = np.squeeze(probabilities)

        ent = entropy(probabilities)

        # with tf.Graph().as_default():
        #     image_string = tf.read_file(image_path)
        #
        #     # Decode string into matrix with intensity values
        #     image = tf.image.decode_jpeg(image_string, channels=3)
        #
        #     # Resize the input image, preserving the aspect ratio and make a central crop of the resulted image.
        #     # The crop will be of the size of the default image size of the network.
        #     processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        #
        #     # Networks accept images in batches. The first dimension usually represents the batch size. In our case the batch size is one.
        #     processed_images = tf.expand_dims(processed_image, 0)
        #
        #     # Create the model, use the default arg scope to configure the batch norm parameters. arg_scope is a very conveniet
        #     # feature of slim library -- you can define default parameters for layers -- like stride, padding etc.
        #     with slim.arg_scope(vgg_arg_scope()):
        #         logits, _ = vgg_19(processed_images, num_classes=1000, is_training=False)
        #
        #     # In order to get probabilities we apply softmax on the output.
        #     probabilities = tf.nn.softmax(logits)
        #
        #     init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_model_dir, 'vgg_19.ckpt'),
        #                                              slim.get_model_variables('vgg_19'))
        #     with tf.Session() as sess:
        #         init_fn(sess)  # load weights
        #
        #         # get predictions, image as numpy matrix and resized and cropped piece that is actually being fed to the network.
        #         np_image, network_input, probabilities = sess.run([image, processed_image, probabilities])
        #         probabilities = probabilities[0, 0:]

        return max(probabilities), ent
    except Exception as e:
        print("Failed to classify image via VGG - " + str(e))
        return 0.0, 0.0

# close session
def CloseVGG(sess):
    try:
        sess.close()
    except Exception as e:
        print("Failed to close VGG - " + str(e))

#######################
# Reduced model inference
#######################
# initialization
def InitializeTransferLearner(pretrained_model_dir, model_name, classes):
    try:
        if(model_name == 'inceptionV3'):
            image_size = inception_v3.default_image_size

            sess = tf.Session()
            input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
            with slim.arg_scope(inception_v3_arg_scope()):
                logits, end_points = inception_v3(input_tensor, num_classes=len(classes), is_training = False)

        elif (model_name == 'inception_resnetV2'):
            image_size = inception_resnet_v2.default_image_size

            sess = tf.Session()
            input_tensor = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(input_tensor, num_classes=len(classes), is_training = False)

        # saver = tf.train.Saver()
        # saver.restore(sess, pretrained_model_dir + 'inception_resnet_v2_2016_08_30.ckpt')

        # # get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)

        ckpt = tf.train.latest_checkpoint(pretrained_model_dir)
        if ckpt:
            # create a saver using variables from the above newly created graph
            saver.restore(sess, ckpt)

        print("Reduced network loaded successfully")
        return(sess, end_points, logits, input_tensor)
    except Exception as e:
        print("Failed to initialize trasnfer learner - " + str(e))

# classification
def ClassifyTransferLearner(sess, end_points, logits, input_tensor, image_path, is_inference = False):
    try:
        image_size = inception_resnet_v2.default_image_size

        processed_image = cv2.imread(image_path)
        processed_image = cv2.resize(processed_image, (image_size, image_size))
        processed_image = processed_image.reshape(-1, image_size, image_size, 3)
        processed_image = 2 * (processed_image / 255.0) - 1.0

        probabilities, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: processed_image})
        probabilities = np.squeeze(probabilities[0])

        pred_class = 'nodule'
        if(probabilities[0] > probabilities[1]):
            pred_class = 'normal'

        if(is_inference):
            actual_class = 'nodule'
            if('normal' in image_path):
                actual_class = 'normal'
        else:
            actual_class = 'normal'

        return max(probabilities), actual_class, pred_class, (actual_class == pred_class), processed_image
    except Exception as e:
        print("Failed to classify image via transfer learner - " + str(e))

# close session
def CloseTransferLearner(sess):
    try:
        sess.close()
    except Exception as e:
        print("Failed to close transfer learner - " + str(e))

# # compute entropy
# def entropy(probabilities):
#     entropy = 0
#     for p in probabilities:
#         p = p / sum(probabilities)
#         if p != 0:
#             entropy += p * log(p, 2)
#         else:
#             entropy += 0
#
#     entropy *= -1
#
#     return entropy