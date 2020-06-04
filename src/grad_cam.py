#==============================================================================
# Author       : Abbas R. Ali
# Last modified: September 04, 2017
# Description  : localization
#==============================================================================

import os
import tensorflow as tf
import numpy as np
from skimage import io
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import torch
import cv2
from torchvision import utils

# import matplotlib
plt.switch_backend('Agg')

plt.ioff()

from src.nets.inception_resnet_v2 import inception_resnet_v2

def grad_cam(raw_image, imgage, input_tensor, end_points, sess, predicted_class, num_classes, output_path):
    try:
        image_size = inception_resnet_v2.default_image_size

        layer_names = ["Mixed_7a", "Predictions"]

        # Conv layer tensor [?,?,?,2048]
        conv_layer = end_points[layer_names[0]]

        # [1000]-D tensor with target class index set to 1 and rest as 0
        one_hot = tf.sparse_to_dense(predicted_class, [num_classes], 1.0)
        signal = tf.multiply(end_points[layer_names[1]], one_hot)
        loss = tf.reduce_mean(signal)

        grads = tf.gradients(loss, conv_layer)[0]

        # Normalizing the gradients
        norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={input_tensor: imgage})
        output = output[0]           # [?,?,2048]
        grads_val = grads_val[0]	 # [?,?,2048]

        weights = np.mean(grads_val, axis = (0, 1)) 			# [2048]
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [?,?]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam3 = cv2.resize(cam, (image_size, image_size))

        # saving image
        imgage = cv2.imread(raw_image)
        imgage = cv2.cvtColor(imgage, cv2.COLOR_BGR2RGB)

        imgage = cv2.resize(imgage, (image_size, image_size))
        imgage = imgage.astype(float)
        imgage /= imgage.max()

        cam3 = cv2.applyColorMap(np.uint8(255 * cam3), cv2.COLORMAP_JET)
        cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

        # Superimposing the visualization with the image.
        alpha = 0.0025
        new_img = imgage + alpha * cam3
        new_img = new_img / new_img.max()

        # cv2.imsave(output_path, new_img)

        # ggcam = gbp * gcam[:, :, np.newaxis]
        # ggcam -= ggcam.min()
        # ggcam = 255 * ggcam / ggcam.max()
        # cv2.imwrite('ggcam.png', ggcam)

        # utils.save_image(torch.from_numpy(new_img), output_path)
        # print('hi')

        # Display and save
        io.imshow(new_img)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

        # plt.show()
    except Exception as e:
        print("Gradiant Cam failed - " + str(e))

# def deprocess_image(x):
#     if np.ndim(x) > 3:
#         x = np.squeeze(x)
#     # normalize tensor: center on 0., ensure std is 0.1
#     x -= x.mean()
#     x /= (x.std() + 1e-5)
#     x *= 0.1
#
#     # clip to [0, 1]
#     x += 0.5
#     x = np.clip(x, 0, 1)
#
#     # convert to RGB array
#     x *= 255
#
#     x = np.clip(x, 0, 255).astype('uint8')
#
#     return x