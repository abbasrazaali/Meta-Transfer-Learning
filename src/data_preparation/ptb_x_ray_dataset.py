#==============================================================================
# Author       : Abbas R. Ali
# Last modified: September 02, 2017
# Description  : prepare chest x-ray dataset for PTB
#==============================================================================

import cv2
import os
from configparser import ConfigParser

def xray_data_preparation(corpus_dir, data_dir, dataset):
    try:
        print("Preparing " + dataset + " data...")

        # training and test folders
        train_path = data_dir + 'train/'
        test_path = data_dir + 'test/'
        # folders = ['CHN', 'MCU']

        count = 0
        for subdir, dirs, files in os.walk(corpus_dir):
            if('labels' in subdir and 'MCU' in subdir or 'CHN' in subdir): #or 'NIH' in subdir):
                for file in files:
                    if (file.endswith('.txt')):
                        with open(subdir + '/' + file, 'r') as readFile:
                            for line in readFile.readlines():
                                label = line.lower().strip()

                            if('normal' in label and len(label) < 10):
                                label = 'normal'
                            else:
                                label = 'nodule'

                            count += 1
                            if(count % 6 == 0):
                                path = test_path
                            else:
                                path = train_path

                            filename = file.split('.')[0]
                            checkPathExists([path + label])

                            os.system('cp ' + os.path.join('/'.join(subdir.split('/')[:-1]), 'images', filename + '.png') + ' ' + os.path.join(path, label, filename + '.png'))

                            if('train' in path):
                                image_orig = cv2.imread(os.path.join(path, label, filename + '.png'))

                                # rotate the image by 180 degrees
                                (h, w) = image_orig.shape[:2]
                                center = (w / 2, h / 2)
                                rotated_img_neg_15 = cv2.warpAffine(image_orig, cv2.getRotationMatrix2D(center, -15, 1.0), (w, h))
                                rotated_img_pos_15 = cv2.warpAffine(image_orig, cv2.getRotationMatrix2D(center, 15, 1.0), (w, h))

                                # horizontal flip
                                horizontal_img = cv2.flip(image_orig, 1)
                                # cv2.imshow("rotated", rotated)
                                # flipped_image = tf.image.flip_left_right(image)

                                cv2.imwrite(os.path.join(path, label, filename + '_flipped.png'), horizontal_img)
                                # cv2.imwrite(os.path.join(path, label, filename + '_rotate_neg.png'), rotated_img_neg_15)
                                cv2.imwrite(os.path.join(path, label, filename + '_rotate_pos.png'), rotated_img_pos_15)
        print(dataset + " data prepared successfully")
    except Exception as e:
        print(dataset + " data data_preparation failed - " + str(e))

# check a path exists or not
def checkPathExists(path):
    for p in path:
        if not os.path.exists(p):
            os.makedirs(p)

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

# main function
def main():
    try:
        main_dir = '../../'

        gConfig = getConfig(main_dir + 'config/metavision.ini')  # get configuration

        dataset = gConfig['datasets']
        corpus_dir = main_dir + gConfig['corpus_dir'] + "/"
        data_dir = gConfig['data_dir'] + "/"

        # dataset = 'ptb'
        # corpus_dir = main_dir + '/Datasets/corpus/' + dataset + '/'
        # data_dir = main_dir + '/Datasets/data/' + dataset + '/'

        xray_data_preparation(corpus_dir + dataset + "/", data_dir + dataset + "/", dataset)

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex

# main function
if __name__ == '__main__':
    main()