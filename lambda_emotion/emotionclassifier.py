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


def load_s3_model(directory_prefix):
    s3 = boto3.client('s3',
                      aws_access_key_id='AKIAZNQEHJKK2XKDESWW',
                      aws_secret_access_key='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',
                      )

    bucket_name = 'pmu-bucket'
    response = s3.list_objects_v2(Bucket=bucket_name)
    directory_prefix = directory_prefix
    local_model_path = '/tmp/'
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory_prefix)

    if 'Contents' in response:
        for item in response['Contents']:
            key_path = item['Key']
            if key_path.endswith('/'):
                continue

            file_name = key_path.split('/')[-1]
            sub_dirs = key_path.split('/')[:-1]

            local_path = os.path.join('content', local_model_path, *sub_dirs)
            if not os.path.exists(local_path):
                os.makedirs(local_path)

            local_file_path = os.path.join(local_path, file_name)
            s3.download_file(bucket_name, key_path, local_file_path)


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
    sys.path.append('/tmp')

    if not os.path.exists('/tmp/model_1'):
        load_s3_model('model_1')
    if not os.path.exists('/tmp/facenet_pytorch'):
        load_s3_model('facenet_pytorch')

    from facenet_pytorch import MTCNN

    model_dir = '/tmp/'
    imageclassifier = ImageClassifier(model_dir)
    pred = imageclassifier.run(event['url'])

    return {
        'emotion': pred
    }