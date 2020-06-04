# ==============================================================================
# Dog vs Cat data preprocessing
# ==============================================================================

import os
import numpy as np
import pickle

from src.utils import *

# dog_cat data preprocessing
def dog_catPreprocessing(dataDir, dataset, logger):
    try:
        # set the file-paths for the caches of the training-set and test-set
        x_train_cache = os.path.join(dataDir, dataset + '_xtrain.pkl')
        x_test_cache = os.path.join(dataDir, dataset + '_xtest.pkl')
        y_train_cache = os.path.join(dataDir, dataset + '_ytrain.pkl')
        y_test_cache = os.path.join(dataDir, dataset + '_ytest.pkl')
        
        logging("Loading training data...", logger, 'info')
         
        # load training data
        X_train, Y_train = loadDog_catTrainingData(dataDir = dataDir, x_train_cache = x_train_cache, y_train_cache = y_train_cache, 
                                                img_size = 50, num_channels = 3, num_classes = 10, logger = logger)

        # load testing data
        X_valid, Y_valid = loadDog_catTestingData(dataDir = dataDir, x_test_cache = x_test_cache, y_test_cache = y_test_cache, 
                                                img_size = 50, num_channels = 3, num_classes = 10, logger = logger)
        
        logging("Training of FC layer has been started", logger, 'info')
        
        return(X_train, Y_train, X_valid, Y_valid)
    except Exception as e:
        logging("Inception multi-layer transfer learning on dog_cat failed - " + str(e), logger, 'error')

# The data-set is split into 5 data-files which are merged here. Returns the images, class-numbers and one-hot encoded class-labels.
def loadDog_catTrainingData(dataDir, x_train_cache, y_train_cache, img_size, num_channels, num_classes, logger = None):
    try:
        if os.path.exists(x_train_cache) and os.path.exists(y_train_cache):
            # load the cached data from the file
            with open(x_train_cache, mode = 'rb') as file:   
                x_train = pickle.load(file)
        
            with open(y_train_cache, mode = 'rb') as file:   
                y_train = pickle.load(file)
        else:
            # pre-allocate the arrays for the images and class-numbers for efficiency.
            x_train = np.zeros(shape = [25000, img_size, img_size, num_channels], dtype = float)
            y_train = np.zeros(shape = [25000, 2], dtype = int)
         
            begin = 0       # begin-index for the current batch
            for i in range(1, 25001):   # for each data-file
                with open(dataDir + 'data_batch_' + str(i), mode = 'rb') as file:        # Load the pickled data-file
                    data = pickle.load(file, encoding = 'bytes')
                
                # Convert images from the CIFAR-10 format and return a 4-dim array with shape: [image_number, height, width, channel]
                # where the pixels are floats between 0.0 and 1.0.
                raw_float = np.array(data[b'data'], dtype = float) / 255.0  # convert the raw images from the data-files to floating-points
                images = raw_float.reshape([-1, num_channels, img_size, img_size])  # reshape the array to 4-dimensions
                images_batch = images.transpose([0, 2, 3, 1])
             
                end = begin + len(images_batch)    # End-index for the current batch
                x_train[begin:end, :] = images_batch     # Store the images into the array
         
                # on hot encoded conversion
                y_train[begin:end] = oneHotEncoded(class_numbers = np.array(data[b'labels']), num_classes = num_classes, logger = logger)
         
                begin = end # The begin-index for the next batch is the current end-index
            
            # cache data
            with open(x_train_cache, mode = 'wb') as file:   
                pickle.dump(x_train, file)
                            
            with open(y_train_cache, mode = 'wb') as file:   
                pickle.dump(y_train, file)
                    
        logging("Dog vs Cat training data loaded successfully.", logger, 'info')
        
        return x_train, y_train
    except Exception as e:
        logging("Failed to load Dog vs Cat training dataset - " + str(e), logger, 'error')

def loadDog_catTestingData(dataDir, x_test_cache, y_test_cache, img_size, num_channels, num_classes, logger = None):
    try:
        if os.path.exists(x_test_cache) and os.path.exists(y_test_cache):
            # load the cached data from the file
            with open(x_test_cache, mode = 'rb') as file:   
                x_test = pickle.load(file)
                
            with open(y_test_cache, mode = 'rb') as file:   
                y_test = pickle.load(file)
        else:
            with open(dataDir + 'test_batch', mode = 'rb') as file:        # Load the pickled data-file
                data = pickle.load(file, encoding = 'bytes')

            # Convert images from the Dog vs Cat format and return a 4-dim array with shape: [image_number, height, width, channel]
            # where the pixels are floats between 0.0 and 1.0.
            raw_float = np.array(data[b'data'], dtype = float) / 255.0  # convert the raw images from the data-files to floating-points
            images = raw_float.reshape([-1, num_channels, img_size, img_size])  # reshape the array to 4-dimensions
            x_test = images.transpose([0, 2, 3, 1])
        
            # on hot encoded conversion
            y_test = oneHotEncoded(class_numbers = np.array(data[b'labels']), num_classes = num_classes, logger = logger)
        
            # cache data
            with open(x_test_cache, mode = 'wb') as file:   
                pickle.dump(x_test, file)
        
            with open(y_test_cache, mode = 'wb') as file:   
                pickle.dump(y_test, file)
                
        logging("Dog vs Cat testing data loaded successfully.", logger, 'info')

        return x_test, y_test
    except Exception as e:
        logging("Failed to load Dog vs Cat testing dataset - " + str(e), logger, 'error')
        