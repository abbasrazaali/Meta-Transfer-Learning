#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 16, 2017
# Description  : prepare chest x-ray dataset
#==============================================================================

import os
import re
from src.utils import getConfig, checkPathExists

def xray_data_preparation(corpus_dir, data_dir, dataset):
    try:
        print("Preparing " + dataset + " data...")

        # training and test folders
        train_path = data_dir + 'train/'
        test_path = data_dir + 'test/'

        labels_files = {}
        test_files = []
        with open(corpus_dir + 'meta/test_list.txt', 'r') as train_file:         # split
            for line in re.split('\r?\n', train_file.read()):
                train_files.append(line.lower())

        with open(corpus_dir + 'meta/test_list.txt', 'r') as test_file:         # split
            for line in re.split('\r?\n', test_file.read()):
                test_files.append(line.lower())

        corpus_dir = corpus_dir + '/images/'
        for label in train_files:       # train data
            if(len(label) > 1):
                label = label.split('/')
                if(os.path.isdir(corpus_dir + label[0])):
                    checkPathExists([train_path + label[0]])
                    os.system('cp ' + os.path.join(corpus_dir, label[0], label[1] + '.jpg') + ' ' + os.path.join(train_path, label[0], label[1] + '.jpg'))

        print("training data prepared successfully")

        for label in test_files:       # test data
            if (len(label) > 1):
                label = label.split('/')
                if(os.path.isdir(corpus_dir + label[0])):
                    checkPathExists([test_path + label[0]])
                    os.system('cp ' + os.path.join(corpus_dir, label[0], label[1] + '.jpg') + ' ' + os.path.join(test_path, label[0], label[1] + '.jpg'))

        print(dataset + " data prepared successfully")
    except Exception as e:
        print(dataset + " data data_preparation failed - " + str(e))

# main function
def main():
    try:
        main_dir = '../../'

        gConfig = getConfig(main_dir + 'config/metavision.ini')  # get configuration

        dataset = gConfig['datasets']
        corpus_dir = main_dir + gConfig['corpus_dir'] + "/"
        data_dir = main_dir + gConfig['data_dir'] + "/"

        xray_data_preparation(corpus_dir + dataset + "/", data_dir + dataset + "/", dataset)

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex

# main function
if __name__ == '__main__':
    main()