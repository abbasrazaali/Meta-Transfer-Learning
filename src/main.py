#==============================================================================
# Author       : Abbas R. Ali
# Last modified: October 16, 2017
# Description  : main function - starting point
#==============================================================================

import os, getopt
import logging as _log
import tensorflow as tf

from src.dataset import dataPreprocessing
from src.inference import InitializeTransferLearner, ClassifyTransferLearner, CloseTransferLearner, InitializeInceptionResnet, ClassifyInceptionResnet, CloseInceptionResnet, InitializeInception, ClassifyInception, CloseInception, ClassifyVGG, InitializeVGG, CloseVGG
from src.transfer_learning import train, test, train_accuracy, test_accuracy, inceptionV3_layers, inceptionResnetV2_layers, vgg_19_layers
from src.grad_cam import grad_cam
from src.deployment.serve import Serving
from src.utils import getConfig, checkPathExists, get_gpu_devices, get_cpu_devices, DeviceCategory, logging

# main function
def main(argv):
    try:
        if (len(argv) > 1):
            if (len(argv) > 1 or argv[1:][0] == '-h'):
                try:
                    opts, args = getopt.getopt(argv[1:], "ho:m:", ["operation=","model="])
                    for opt, arg in opts:
                        opt = opt.lower()
                        arg = arg.lower()
                        if opt == '-h':
                            print('voicenet.py -o <train|test|infer|analysis|serve> -m <inceptionV3,inception_resnetV2,vgg_19>')
                            return
                        elif opt in ("-o", "--operation"):
                            mode = arg
                        elif opt in ("-m", "--model"):
                            pretrained_models = arg

                except getopt.GetoptError:
                    print('voicenet.py -o <train|test|infer|analysis|serve> -m <inceptionV3,inception_resnetV2,vgg_19>')  # -o <data_prep|train_test|freeze_model|infer|serve|regress_infer|analysis>')
                    return

                if pretrained_models in 'inceptionV3' or pretrained_models in 'inception':
                    pretrained_models = 'inceptionV3'
                elif pretrained_models in 'inception_resnetV2' or pretrained_models in 'resnet':
                    pretrained_models = 'inception_resnetV2'
                elif pretrained_models in 'vgg_19' or pretrained_models in 'vgg':
                    pretrained_models = 'vgg_19'
        else:
            mode = ''
            pretrained_models = ''

        # if len(argv):
        #     gConfig = getConfig(main_dir + 'config/' + getConfig(argv[1]).lower() + '.ini')   # get configuration
        # else:
        gConfig = getConfig('config/metavision.ini')  # get configuration

        site = gConfig['site']
        if(not len(mode)):
            mode = gConfig['mode']

        if (not len(pretrained_models)):
            pretrained_models = gConfig['pretrained_model_dir']

        datasets = gConfig['datasets']
        data_dirs = gConfig['data_dir']
        infer_dir = gConfig['infer_dir'] + "/" + datasets + "/"
        train_num_epochs = gConfig['train_num_epochs']
        test_num_epochs = gConfig['test_num_epochs']
        layer_start = gConfig['layer_start']
        infer_layer = gConfig['infer_layer']
        learning_rate = gConfig['learning_rate']
        learning_rate_decay_factor = gConfig['learning_rate_decay_factor']
        num_epochs_per_decay = gConfig['num_epochs_per_decay']
        train_batch_size = gConfig['train_batch_size']
        test_batch_size = gConfig['test_batch_size']
        optimizer = gConfig['optimizer']
        dropout_keep_prob = gConfig['dropout_keep_prob']
        extract_features_only = gConfig['extract_features_only']
        log_dir = gConfig['log_dir']
        port = gConfig['port']
        gpu_to_use = gConfig['gpu_to_use']
        certificate = gConfig['certificate']
        resource_dir = gConfig['resources']

        # init_inception = False
        # init_inception_resnet = False
        # init_vgg = False
        #
        # logits = None

        # create logger
        _log.basicConfig(filename=log_dir + "/" + "log.txt", level=_log.DEBUG, format='%(asctime)s %(message)s',
                         datefmt='%m/%d/%Y %I:%M:%S %p')
        logger = _log.getLogger("VoiceNet")
        logger.setLevel(_log.DEBUG)
        console = _log.StreamHandler()
        console.setLevel(_log.DEBUG)

        formatter = _log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")  # create formatter
        console.setFormatter(formatter)
        logger.addHandler(console)

        if ('train' in mode or 'test' in mode):
            # specify GPU numbers to use get gpu and cpu devices
            cpu_devices = get_cpu_devices()
            gpu_devices = get_gpu_devices()
            if (len(gpu_devices) > 1):
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gConfig["gpu_to_use"])

                print("The available GPU devices: " + str(gpu_devices))

                # devices, device_category = (gpu_devices, DeviceCategory.GPU) if len(gpu_devices) > 1 else (cpu_devices, DeviceCategory.CPU)

            for pretrained_model in pretrained_models.split(','):     # pre-trained architecture
                pretrained_model_dir = "pretrained_models/" + pretrained_model + "/"
                for dataset in datasets.split(','):     # datasets
                    data_dir = data_dirs + "/" + dataset + "/"
                    features_dir = gConfig['features_dir'] + "/" + pretrained_model + "/" + dataset + "/"
                    model_dir = gConfig['model_dir'] + "/" + pretrained_model + "/" + dataset + "/"
                    log_dir = gConfig['log_dir'] + "/" + pretrained_model + "/" # + dataset + "/"
                    output_dir = gConfig['output_dir'] + "/" + pretrained_model + "/" + dataset + "/"

                    checkPathExists([model_dir, features_dir, log_dir, output_dir])

                    if (pretrained_model == 'inceptionV3'):         # inception v3
                        architecture_layers = inceptionV3_layers
                    elif (pretrained_model == 'inception_resnetV2'):        # inception_resnet v2
                        architecture_layers = inceptionResnetV2_layers
                    elif (pretrained_model == 'vgg_19'):        # vgg 19
                        architecture_layers = vgg_19_layers

                    logging(pretrained_model + " Transfer learning on " + dataset + " dataset", logger, 'info')

                    if(pretrained_model == 'vgg_19'):
                        layer_count_len = len(architecture_layers)
                    else:
                        layer_count_len = len(architecture_layers) - 1

                    if (extract_features_only):
                        layer_count_len = 2

                    for layer_count in range(layer_start, layer_count_len):
                        # learning_rate_ = round((learning_rate / (1.05 + ((layer_count - 1) / 10))), 8)
                        train_num_epochs_ = train_num_epochs + (10 * (layer_count - 1))

                        if("train" in mode):        # training
                            with tf.Graph().as_default() as graph:
                                tf.logging.set_verbosity(tf.logging.ERROR)  # set the verbosity to INFO level

                                train_dataset, train_images, train_labels = dataPreprocessing('train', data_dir, features_dir, train_batch_size, pretrained_model, logger)

                                if(not extract_features_only):
                                    train_accuracy = train('train', train_dataset, train_images, train_labels, train_batch_size, train_num_epochs_,
                                          optimizer, learning_rate, learning_rate_decay_factor, num_epochs_per_decay, dropout_keep_prob,
                                          pretrained_model, model_dir, pretrained_model_dir, layer_count, logger)

                                graph.finalize()

                        if ("test" in mode):        # testing
                            with tf.Graph().as_default() as graph:
                                tf.logging.set_verbosity(tf.logging.ERROR)  # set the verbosity to INFO level

                                test_dataset, test_images, test_labels = dataPreprocessing('test', data_dir, features_dir, test_batch_size, pretrained_model, logger)

                                if (not extract_features_only):
                                    test_accuracy = test('test', test_dataset, test_images, test_labels, test_batch_size, test_num_epochs, pretrained_model, model_dir, layer_count, logger)
                                    # test_accuracy = test('test', test_dataset, data_dir, pretrained_model, model_dir, layer_count, logger)

                                graph.finalize()

                    if (not extract_features_only):
                        # save the results file
                        logging(pretrained_model + " Writing results of " + dataset + " dataset", logger, 'info')
                        if ("train" in mode):  # training
                            with open(output_dir + 'accuracy.txt', 'w') as writeresultsDict:
                                writeresultsDict.write('Architecture\tDataset\tLayer\tEpochs\tLearning\tTrain_Loss\tTrain_Accuracy\tMax_Train_Accuracy\n')
                                # for train_k, train_v in train_accuracy.items():
                                for layer_count in range(1, layer_count_len):
                                    print(train_accuracy[str(layer_count)])
                                    # accuracy = train_accuracy[layer_count].split('\t')
                                    writeresultsDict.write(pretrained_model + '\t' + dataset + '\t' + str(layer_count) + '\t' + train_accuracy[str(layer_count)] + '\n')

                        if ("test" in mode):  # testing
                            with open(output_dir + 'test_accuracy.txt', 'w') as writeresultsDict:
                                writeresultsDict.write('Architecture\tDataset\tLayer\tTest_Accuracy\n')
                                # for test_k, test_v in test_accuracy.items():
                                for layer_count in range(1, layer_count_len):
                                    # accuracy = test_accuracy[layer_count].split('\t')
                                    writeresultsDict.write(pretrained_model + '\t' + dataset + '\t' + str(layer_count) + '\t' + str(test_accuracy[str(layer_count)]) + '\n')

        elif 'infer' in mode:
            output_dir = gConfig['output_dir'] + "/" + pretrained_models + "/" + datasets + "/"
            model_dir = gConfig['model_dir'] + "/" + pretrained_models + "/" + datasets + "/" + str(infer_layer) + "/"
            checkPathExists([output_dir, model_dir])

            inferenceResults = open(gConfig['output_dir'] + '/inference.txt', 'w')
            inferenceResults.write('Architecture\tActual Class\tPredicted Class\tProbability\n')
            # test image
            # image = "willy_wonka_new.jpg"

            # preprocessing
            # input_tensor, _, _ = dataPreprocessing('infer', infer_dir, dataset, train_batch_size, pretrained_model, logger)

            # inference
            # predictions = inference(mode, pretrained_model, pretrained_model_dir, infer_dir + imagefile, channels = 3, return_top_predictions=5)
            # PlotResizedImage(sess, image_path=image_path)
            # ineption_prediction = ClassifyInception(sess, image_path, return_top_predictions=5)

            # print(pretrained_model + ' - network prediction: ' + str(predictions) + '\n')

            classes = []
            with open(resource_dir + '/' + datasets + '/labels.txt', 'r') as readfile:
                for line in readfile.readlines():
                    classes.append(line.split(':')[1].strip())

            sess_transfer_learner, end_points, logits, input_tensor = InitializeTransferLearner(model_dir, pretrained_models, classes)
            init_model = True

            # # initialization
            # if (pretrained_model == 'inceptionV3' and not init_inception):  # inception v3
            #     sess_inception = InitializeInception(pretrained_model_dir)
            #     init_inception = True
            # elif (pretrained_model == 'inception_resnetV2' and not init_inception_resnet):  # inception_resnet v2
            #     sess_inception_resnet, end_points, logits, input_tensor, imagenet_classes = InitializeInceptionResnet(model_dir)
            #     init_inception_resnet = True
            # elif (pretrained_model == 'vgg_19' and not init_vgg):  # vgg 19
            #     sess_vgg, prediction, input_tensor = InitializeVGG(pretrained_model_dir)
            #     init_vgg = True

            # print('Inception - Resnet network prediction: ' + str(ineption_resnet_prediction[0]) + '\n')

            logging(datasets + " inference on " + pretrained_models + " network", logger, 'info')

            # inference
            count = 0
            probability = 0.0
            accuracy_  = 0.0
            # entropy = 0.0
            for subdir, dirs, files in os.walk(os.path.join(infer_dir)):
                for file in files:
                    if file.endswith('.png') or file.endswith('.jpg'):
                        # if (pretrained_model == 'inceptionV3' and init_inception):  # inception v3
                        #     probabilities, entropies = ClassifyInception(sess_inception, subdir + "/" + file)
                        # elif (pretrained_model == 'inception_resnetV2' and init_inception_resnet):  # inception_resnet v2
                        #     probabilities, entropies = ClassifyInceptionResnet(sess_inception_resnet, end_points, logits, input_tensor, subdir + "/" + file)
                        # elif (pretrained_model == 'vgg_19' and init_vgg):  # vgg 19
                        #     probabilities, entropies = ClassifyVGG(sess_vgg, prediction, input_tensor, subdir + "/" + file)

                        # if(init_model):
                        probabilities, actual_class, pred_class, accuracy, processed_image = ClassifyTransferLearner(sess_transfer_learner, end_points, logits, input_tensor, subdir + "/" + file, is_inference=True)

                        grad_cam(subdir + "/" + file, processed_image, input_tensor, end_points, sess_transfer_learner, classes.index(pred_class), num_classes = len(classes),
                                 output_path = subdir + "/" + file.split('.')[0] + '_cam.jpg')

                        probability += probabilities
                        accuracy_ += accuracy

                        inferenceResults.write(pretrained_models + '\t' + str(file) + '\t' + actual_class + '\t' + pred_class + '\t' + str(accuracy) + '\t' + str(round(probabilities, 2)) + '\n')

                        count += 1

            if(count):
                probability = (probability * 100) / count
                accuracy_ = (accuracy_ * 100) / count

            inferenceResults.write(pretrained_models + '\t' "Accuracy: " + str(accuracy_) + '\t' + "Probability: " + str(round(probability, 2)) + '\n')

            print(pretrained_models + ' network predictions on ' + datasets + " - Probability: " + str(probability) + " - Accuracy: " + str(accuracy_))

            # if (pretrained_model == 'inceptionV3' and init_inception):  # inception v3
            #     CloseInceptionResnet(sess_inception)
            #     init_inception = False
            # elif (pretrained_model == 'inception_resnetV2' and init_inception_resnet):  # inception_resnet v2
            #     CloseInceptionResnet(sess_inception_resnet)
            #     init_inception_resnet = False
            # elif (pretrained_model == 'vgg_19' and init_vgg):  # vgg 19
            #     CloseVGG(sess_vgg)
            #     init_vgg = False

            CloseTransferLearner(sess_transfer_learner)
            inferenceResults.close()
        elif (mode == "serve"):  # serve
            output_dir = gConfig['output_dir'] + "/" + pretrained_models + "/" + datasets + "/"
            model_dir = gConfig['model_dir'] + "/" + pretrained_models + "/" + datasets + "/" + str(infer_layer) + "/"
            checkPathExists([output_dir, model_dir])

            classes = []
            with open(resource_dir + '/' + datasets + '/labels.txt', 'r') as readfile:
                for line in readfile.readlines():
                    classes.append(line.split(':')[1].strip())

            model_server = Serving(site, port, model_dir, pretrained_models, infer_dir, output_dir, log_dir, gpu_to_use, classes, certificate=certificate, logger=logger)

            model_server.run()

    except Exception as ex:
        print("main function failed - " + str(ex))
        raise ex