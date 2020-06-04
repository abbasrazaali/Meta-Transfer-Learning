#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 16, 2017
# Description  : prepare caltech dataset
#==============================================================================

import os
from src.utils import getConfig, checkPathExists

def caltech_data_preparation(corpus_dir, data_dir, dataset, testset_proportion):
    try:
        print("Preparing " + dataset + " data...")

        labels = os.listdir(corpus_dir)       # get all the images and labels in directory

        # training and test folders
        train_path = data_dir + 'train/'
        test_path = data_dir + 'test/'

        for label in labels:
            if(os.path.isdir(corpus_dir + label)):
                checkPathExists([train_path + label])
                checkPathExists([test_path + label])
                count = 0
                total_files = len(os.listdir(corpus_dir + label))
                test_file = int(total_files / (total_files * testset_proportion / 100))
                for filename in os.listdir(corpus_dir + label):
                    if(count % test_file != 0):
                        os.system('cp ' + os.path.join(corpus_dir, label, filename) + ' ' + os.path.join(train_path, label, filename))
                    else:
                        os.system('cp ' + os.path.join(corpus_dir, label, filename) + ' ' + os.path.join(test_path, label, filename))
                    count += 1

        print(dataset + " data prepared successfully")
    except Exception as e:
        print(dataset + " data data_preparation failed - " + str(e))

# main function
def main():
    try:
        main_dir = '../'

        gConfig = getConfig(main_dir + 'config/metavision.ini')  # get configuration

        dataset = 'caltech'
        corpus_dir = main_dir + gConfig['corpus_dir'] + "/"
        data_dir = main_dir + gConfig['data_dir'] + "/"
        testset_proportion = gConfig['testset_proportion']

        caltech_data_preparation(corpus_dir + dataset + "/", data_dir + dataset + "/", dataset, testset_proportion)

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex

# main function
if __name__ == '__main__':
    main()