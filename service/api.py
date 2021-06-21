from argparse import ArgumentParser
import base64
import datetime
import hashlib
import traceback
import cv2
import os

from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
import tensorflow as tf
import json
import time
import my_logger

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'u9652018@gmail.com'          #
SALT = 'nick_ccc'                        #
#########################################

class Model:
    def __init__(self, efficientnet_v1_model_path,
                       sam_model_path,
                       label_map_file_path):

        self.efficientnet_v1_model_path = os.path.join(os.getcwd(), efficientnet_v1_model_path)
        self.sam_model_path = os.path.join(os.getcwd(), sam_model_path)
        self.label_path = label_map_file_path

        self.load_model()
        self.load_label_map()
        
    def load_model(self):
        self.efficientnet_v1_model = tf.keras.models.load_model(self.efficientnet_v1_model_path, compile=False)
        self.sam_model = tf.keras.models.load_model(self.sam_model_path, compile=False)

    def load_label_map(self):
        with open(self.label_path, 'r', encoding='utf-8') as json_file:
            self.ans2idx = json.load(json_file)

        self.idx2ans = {v: k for k, v in self.ans2idx.items()}

    def predict(self, image):

        img_array_rgb = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        img_array_rgb = tf.expand_dims(img_array_rgb, 0)
        
        image_cv = cv2.GaussianBlur(image, (3, 3), 0)
        image_cv = cv2.Canny(image_cv, 150, 200)
        image_cv = cv2.resize(image_cv, (224, 224), interpolation=cv2.INTER_CUBIC)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)
        img_array_cv = tf.expand_dims(image_cv, 0)

        images = tf.keras.layers.concatenate([img_array_rgb, img_array_cv], axis=0)
        images = tf.cast(images, tf.float32)

        efficientnet_v1_predict = self.efficientnet_v1_model.predict(images)
        ind = np.unravel_index(np.argmax(efficientnet_v1_predict, axis=None), efficientnet_v1_predict.shape)
        confidence_eff_v1 = efficientnet_v1_predict[ind]
        ans = ind[1]


        sam_predict = self.sam_model.predict(images)
        ind = np.unravel_index(np.argmax(sam_predict, axis=None), sam_predict.shape)
        confidence_sam = sam_predict[ind]

        if ind[1] == 0:
            return self.idx2ans[0]

        if confidence_sam > confidence_eff_v1:
            ans = ind[1]


        return self.idx2ans[ans]

def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    # image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_GRAYSCALE)

    return image


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######




    prediction = '陳'


    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """

    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)
    answer = 'isnull'
    try:
        # answer = predict(image)
        answer = model.predict(image)
        inference_logger.debug('data: {}'.format(data))

    except TypeError as type_error:
        # You can write some log...
        inference_logger.debug('type_error')
        # raise type_error
    except Exception as e:
        # You can write some log...
        inference_logger.debug('Exception: {}'.format(e))
        inference_logger.error('predict error: {}'.format(traceback.print_exc()))
        # raise e
    # server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    server_timestamp = time.time()
    try:
        return_json = {'esun_uuid': data['esun_uuid'],
                       'server_uuid': server_uuid,
                       'answer': answer,
                       'server_timestamp': server_timestamp}
        inference_logger.debug('return: {}'.format(return_json))
    except Exception as e:
        inference_logger.debug('Exception (return_json): {}'.format(e))

    return jsonify(return_json)


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    arg_parser.add_argument('-l', '--log_path', default="log.log", help='log_path')
    options = arg_parser.parse_args()
    inference_logger = my_logger.set_logger('inference', logPath=f"{options.port}_log.log")
    efficientnet_v1_model_path = './fine_tuned_model/efficient_v1.hdf5'
    sam_model_path = './fine_tuned_model/efficient_SAM.hdf5'
    label_map_file_path = 'label_map.json'
    model = Model(efficientnet_v1_model_path,
                  sam_model_path,
                  label_map_file_path)
    test_img = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCABDACYDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDuf7I8a+eJjdQ7sY5k/wDsD/WnvYeOsEC6g/PP+H8q7xysaF2YBRySegpkFxBcReZC6yIehHQ4/wD1Vj7Bd2dzzCb+xH7jhBZ+OQObmGl+zeOx0nTH/AK6eTVZX8Rx6dbIjRqm+d/7vcfpWxx1GMHpR9XS6sn69L+SP3Hn5tvHv8Mif99pRXoA4oo9gu7F9dl/JH7v+CZ+vSeR4fvpP7sD/wAqxfDjJonge3mdN2IzKMfxFiSB+tTeP7tbTwbqDE43JtqXQYYtT8F2kEvKTW4jP5V1JL2bvtc4tLlBZr7QbF9QFg99JdN5kzRnLAHoMewwPwq9ovivTNbmaCAtDOp5gmXBpnhi4uYjdaPeSZmtJMRt6x8Yo8TeHItUga8tsQ6lCC0E46g//Xqpcrdpr5/kB0q8jv8AjRWJ4Y1c6voySOhW4jPlzA/3xwfzxn8aKxa5Xyy3GY/jdDqT2GixjD3Mm5v9wHn+VS+A3Kabdac33rG5aP8AA8j9DU+mxLe+Lr+/dywtgIIx2zgZ/XNVfDKfZ/FWvW+ACXSTA/Cuh6UvZ9tSba3M/wAQahc6L47tpbW1e7lvLcqIl+X25NdJpdxrs5zqVjb28Z24CS7iOaxtSdV+JulgjcVtmGPTJJrs+DwfUGoqP3Y6dBpnIeF0+y+KNesQcosglGP9qil8NZuPFGvXKr8nmKg/AUVVTcR0ljp9vp0PlWy7VZ97E+prB0WBl8aa1I5yWVNv5CupzxgdPaq0NnDb3E10qYllxuP0AH9KyUtNepWhhpotxc+Ibq+u9ojVdsDd14HP51Dd6rr0NtJZppjzXTEok4+6y/3q6vGAAKN2OKam3uGhkeHdI/snSlgYBpXYvIw/vEk4/WitffRUuTb2A4KPWtRKf8fLf98j/CrEOq3zr81y/wCGBRRUM6x51O9HS5eo21jUExtuWH4D/CiikaUkm2N/tzUv+fpv++R/hRRRSNuSPY//2Q=="
    test_img = base64_to_binary_for_cv2(test_img)
    model.predict(test_img)
    del test_img
    app.run(host='0.0.0.0', debug=options.debug, port=options.port)
