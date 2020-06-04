#==============================================================================
# Author       : Abbas R. Ali
# Last modified: September 4, 2018
# Description  : meta-vision serving
#==============================================================================

import os
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template, Response
from base64 import b64encode
from json import dumps
from flask_cors import CORS, cross_origin

from src.grad_cam import grad_cam
from src.inference import InitializeTransferLearner, ClassifyTransferLearner
from src.utils import checkPathExists, get_gpu_devices, get_cpu_devices, DeviceCategory, logging

APP = Flask(__name__)
CORS(APP, support_credentials=True)

class Serving():
    def __init__(self, site, port, model_dir, pretrained_model, infer_dir, output_dir, log_dir, gpu_to_use, classes, certificate=None, logger=None):
        self.model_dir = model_dir
        self.log_dir = log_dir + '/' + site + '/'
        self.output_dir = output_dir
        self.port = port
        self.infer_dir = infer_dir + '/' + site + '/'
        self.pretrained_model = pretrained_model
        self.certificate = certificate
        self.classes = classes
        self.logger = logger

        self.log_filename = None

        self.sess_transfer_learner = None
        self.end_points = None
        self.logits = None
        self.input_tensor = None

        self.initization = False

        self.devices = None
        self.gpu_to_use = gpu_to_use

        self.init_app_blue_print()

    def init_app_blue_print(self):
        @APP.before_first_request
        def load_model():
            # setting gpus
            self.cpu_devices = get_cpu_devices()
            self.gpu_devices = get_gpu_devices()
            if (len(self.gpu_devices) > 0):
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_to_use)

            self.devices, _ = (self.gpu_devices, DeviceCategory.GPU) if len(self.gpu_devices) > 0 else (self.cpu_devices, DeviceCategory.CPU)

            # initialization
            self.sess_transfer_learner, self.end_points, self.logits, self.input_tensor = InitializeTransferLearner(self.model_dir, self.pretrained_model, self.classes)
            init_model = True

            # logging
            checkPathExists([self.infer_dir, self.log_dir])
            self.log_filename = self.log_dir + '/' + datetime.now().strftime('prediction_logs_%H_%M_%d_%m_%Y.log')
            with open(self.log_filename, 'w') as prediction_log:
                prediction_log.write('Filename' + '\t' + 'Actual' + '\t' + 'Predicted' + '\t' + 'Score' + '\t' + 'Noise_Suppression' + '\t' + 'Language_Model' + '\n')

            logging('Inference engine has been initiated on port: ' + str(self.port), self.logger, 'info')

            self.initization = True

        @APP.route('/', methods=['GET', 'POST'])
        @cross_origin(supports_credentials=True)
        def index():
            if request.method == 'POST' and self.initization:
                file = request.files['query_img']

                image = Image.open(file.stream)  # PIL image
                # uploaded_img_path = os.path.abspath(self.infer_dir + file.filename)         # uploaded_img_path = os.path.abspath(self.infer_dir + file.filename)

                uploaded_img_path_orig = "src/deployment/static/uploaded/" + file.filename
                uploaded_img_path_orig_cam = uploaded_img_path_orig[:-4] + '_cam.png'
                image.save(uploaded_img_path_orig)

                probabilities, _, pred_class, _, processed_image = ClassifyTransferLearner(self.sess_transfer_learner,
                                                                    self.end_points, self.logits, self.input_tensor, uploaded_img_path_orig)

                # uploaded_img_path = '/' + '/'.join(uploaded_img_path_orig.split('/')[2:])
                # uploaded_img_path_cam = uploaded_img_path[:-4] + '_cam.png'

                if('normal' in pred_class):
                    # results = [pred_class + '\t|\tConfidence level: ' + str(round(probabilities, 2)), None]
                    results = dumps({
                        'pred_class' :  pred_class + '\t|\tConfidence level: ' + str(round(probabilities, 2))
                    })
                else:
                    grad_cam(uploaded_img_path_orig, processed_image, self.input_tensor, self.end_points, self.sess_transfer_learner,
                             self.classes.index(pred_class), num_classes=len(self.classes), output_path=uploaded_img_path_orig_cam)

                    bytes_string = b64encode(open(uploaded_img_path_orig_cam, 'rb').read()).decode('utf-8')
                    # bytes_string = bytes_content.decode('utf-8')
                    # bytes_string = bytes_content
                    #results = [pred_class + '\t|\tConfidence level: ' + str(round(probabilities, 2)),  ]
                    results = dumps({
                        'pred_class' :  pred_class + '\t|\tConfidence level: ' + str(round(probabilities, 2)),
                        'image_output' : bytes_string
                    })
                # print(uploaded_img_path_cam)
                # print(pred_class)
                # print(str(round(probabilities, 2)))
                # uploaded_img_path_orig = b64encode(open(uploaded_img_path_orig, 'rb').read())
                # scores_path = scores[0]
                # scores = scores[1]
                # response = {'query_path': uploaded_img_path_orig, 'scores_path': scores[0], 'scores': uploaded_img_path_orig}
                #response = {'cam_image_path': results[1], 'score': results[0]}


                # return render_template('index.html', query_path=uploaded_img_path, scores_path=scores_path, scores=scores)
                # return render_template('index.html', query_path=uploaded_img_path)
                # response = response.to_rest
                # response = json.dumps(response)
                # response = flask.jsonify(response)
                response = Response(response=results) #, mimetype='application/json')
                # response.headers['Access-Control-Allow-Origin'] = '*'
                return response
            else:
                return render_template('index.html')

    def run(self):
        with APP.app_context():
            if ('None' not in self.certificate):
                APP.run(debug=False, host='0.0.0.0', port=self.port, ssl_context=(self.certificate + '/cert.pem', self.certificate + '/key.pem'))
            else:
                APP.run(debug=True, host='0.0.0.0', port=self.port)
