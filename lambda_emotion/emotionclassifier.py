from PIL import Image
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend as K
import requests
import io
import boto3

sys.path.append('/var/task')
from facenet_pytorch import MTCNN


class ImageClassifier:

    def __init__(self, model_dir):
        self.model = load_model(os.path.join(model_dir, 'model_1'),
                                custom_objects={'get_f1': self.get_f1})
        self.mtcnn = MTCNN(select_largest=False, post_process=False)
        self.url = ""
        self.image_size = {'IMG_HEIGHT': 128, 'IMG_WIDTH': 128}
        self.labels = ['neutral', 'sad', 'happy', 'angry']

    def get_f1(y_true, y_pred):

        '''
        custom metric when we used in model
        '''
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    def url_to_input(self):

        '''
        Image preProcessing
        parameters : Profile Image of KaKao
        return : preprocessed Image

        '''

        if self.url:
            response = requests.get(self.url)
            if response.status_code == 200:
                img_data = response.content
                image = Image.open(io.BytesIO(img_data))
                image = image.convert("RGB")

                image = self.mtcnn(image)
                image = np.array(image.permute(1, 2, 0).int())

        image_np = np.array(image) / 255.
        image_tensor = tf.convert_to_tensor(image_np)

        image_resized = tf.image.resize(image_tensor, (self.image_size['IMG_HEIGHT'],
                                                       self.image_size['IMG_WIDTH']))

        image = np.expand_dims(image_resized, axis=0)

        return image

    def predict(self, image):
        pred = self.model.predict(image)
        pred = np.argmax(pred, axis=1)

        return self.labels[pred.item()]

    def run(self, url: str):

        self.url = url
        image = self.url_to_input()
        pred = self.predict(image)
        return pred


def lambda_handler(event, context=None, url=""):
    '''

    lambda_handler argument -> event: json
    '''

    model_dir = '/var/task'

    imageclassifier = ImageClassifier(model_dir)
    pred = imageclassifier.run(event['url'])

    return {
        'emotion': pred
    }