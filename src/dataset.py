#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 16, 2017
# Description  : dataset data_preparation and preprocessing
#==============================================================================

import os
# import glob
import re
import tensorflow as tf
from src.preprocessing import inception_preprocessing
from src.preprocessing import vgg_preprocessing
from src.nets.inception_v3 import inception_v3
from src.nets.inception_resnet_v2 import inception_resnet_v2
from src.nets.vgg import vgg_19
from src.utils import logging   #, oneHotEncoded, reshapeImage

slim = tf.contrib.slim

# classes
labels_dict = {}

# data preprocessing
def dataPreprocessing(mode, data_dir, features_dir, batch_size, pretrained_model, logger = None):
    try:
        labels_dict = {}  # label dictionary

        if (not os.path.isfile(features_dir + "/" + mode + ".tfrecord")):       # check whether feature file already exist
            logging("Preparing " + mode + "ing data...", logger, 'info')

            record_writer = tf.python_io.TFRecordWriter(path=features_dir + "/" + mode + ".tfrecord")

            labels = os.listdir(data_dir + "/" + mode)  # get all the images and labels in directory
            labels.sort()  # sort the labels so that training and validation get them in the same order

            if (not os.path.isfile(features_dir + '/labels.txt')):  # check whether feature file already exist
                for i, label in enumerate(list(set(labels))):           # preparing labels
                    labels_dict[label] = i

                if (not os.path.isfile(features_dir + '/labels.txt')):  # check whether labels file already exist
                    with open(features_dir + '/labels.txt', 'w') as writelabelDict:  # write labels file
                        for k in sorted(labels_dict, key=labels_dict.get):
                            writelabelDict.write(str(labels_dict[k]) + ':' + k + '\n')
            else:
                with open(features_dir + '/labels.txt', 'r') as label_file:  # load labels
                    for line in re.split('\r?\n', label_file.read()):
                        line = line.split(':')
                        if len(line[0]) and len(line[1].strip()) and not line[1].strip() in labels_dict:
                            labels_dict[line[1].strip().lower()] = int(line[0])

            image_reader = ImageReader()
            with tf.Session('') as sess:
                for label in labels:
                    for filename in os.listdir(os.path.join(data_dir, mode, label)):
                        if(filename not in '.DS_Store'):
                            image_data = tf.gfile.FastGFile(os.path.join(data_dir,  mode, label, filename), 'rb').read()          # extract image features
                            height, width = image_reader.read_image_dims(sess, image_data)

                            example =  tf.train.Example(features=tf.train.Features(feature={            # tensorflow example
                                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                                'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
                                'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels_dict[label]])),
                                'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                                'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                            }))

                            record_writer.write(example.SerializeToString())

                record_writer.flush()
                record_writer.close()

        # load dataset
        with open(features_dir + '/labels.txt', 'r') as label_file:         # load labels
            for line in re.split('\r?\n', label_file.read()):
                line = line.split(':')
                if len(line[0]) and len(line[1].strip()) and not line[1].strip() in labels_dict:
                    labels_dict[int(line[0])] = line[1].strip().lower()

        num_samples = 0         # Count the total number of examples in all of these shard
        for _ in tf.python_io.tf_record_iterator(features_dir + "/" + mode + ".tfrecord"): num_samples += 1

        reader = tf.TFRecordReader          # create a reader, which must be a TFRecord reader in this case

        # create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        # create the items_to_handlers dictionary for the decoder.
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        # start to create the decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        # create the dataset
        dataset = slim.dataset.Dataset(
            data_sources=features_dir + "/" + mode + ".tfrecord",
            decoder=decoder,
            reader=reader,
            num_readers=4,
            num_samples=num_samples,
            num_classes=len(labels_dict),
            labels_to_name=labels_dict,
            items_to_descriptions={})

        # create the data_provider object
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=24 + 3 * batch_size,
            common_queue_min=24)

        # obtain the raw image using the get method
        raw_image, label = data_provider.get(['image', 'label'])

        # perform the correct preprocessing for this image depending if it is training or evaluating
        if(pretrained_model == 'inceptionV3'):          # inception v3
            image_size = inception_v3.default_image_size
            image = inception_preprocessing.preprocess_image(raw_image, height = image_size, width = image_size, is_training = (mode == 'train'))
        elif(pretrained_model == 'inception_resnetV2'):         # inception_resnet v2
            image_size = inception_resnet_v2.default_image_size
            image = inception_preprocessing.preprocess_image(raw_image, height = image_size, width = image_size, is_training = (mode == 'train'))
        elif (pretrained_model == 'vgg_19'):  # vgg 19
            image_size = vgg_19.default_image_size
            image = vgg_preprocessing.preprocess_image(raw_image, output_height = image_size, output_width = image_size, is_training = (mode == 'train'))

        # as for the raw images reshape to batch it up
        # raw_image = tf.expand_dims(raw_image, 0)
        # raw_image = tf.image.resize_nearest_neighbor(raw_image, [image_size, image_size])
        # raw_image = tf.squeeze(raw_image)

        # batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=4 * batch_size,
            allow_smaller_final_batch=True)

        logging(mode + "ing data loaded successfully", logger, 'info')

        return dataset, images, labels
    except Exception as e:
        logging("Data preprocessing failed - " + str(e), logger, 'error')

# create an image reader object for easy reading of the images, class that provides TensorFlow image coding utilities
class ImageReader(object):
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image