
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend as K




class ImageClassifier:

  def __init__(self,model_dir : str,
               ) -> None:
    self.model = load_model(model_dir,
                            custom_objects={'get_f1': self.get_f1})
    self.image = ""

    self.image_size = {'IMG_HEIGHT' : 128, 'IMG_WIDTH' : 128}

    self.labels=['neutral', 'sad', 'happy', 'angry']

  def get_f1(y_true, y_pred):

    '''
    custom metric when we used in model
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

  def image_to_input(self):

    '''
    Image preProcessing
    parameters : Profile Image of KaKao
    return : preprocessed Image

    '''
    image = Image.open(os.path.join(f'{self.image}'))

    image_np = np.array(image) / 255.
    image_tensor = tf.convert_to_tensor(image_np)

    image_resized = tf.image.resize(image_tensor, (self.image_size['IMG_HEIGHT'],
                                                   self.image_size['IMG_WIDTH']))

    image = np.expand_dims(image_resized, axis=0)


    return image



  def predict(self, image):
    pred=self.model.predict(image)
    pred=np.argmax(pred,axis=1)

    return self.labels[pred.item()]


  def run(self,image) :
    self.image=image
    image=self.image_to_input()
    pred=self.predict(image)
    return pred

# model_dir은 github/dir
imageclassifier=ImageClassifier(
    model_dir = "/var/task/model_1",
)


def lambda_handler(event=None, context=None,url=""):
    '''
    lambda_handler로 들어오는 이미지 url을 sample_img에 담을거임.
    url로 수정해야함 . 지금은 그냥 샘플이미지
    '''
    prediction=[]
    sample_img1 = "/var/task/sample_1.jpg"
    sample_img2 = "/var/task/sample_2.jpg"
    sample_img3 = "/var/task/sample_3.jpg"
    sample_img4 = "/var/task/sample_4.jpg"
    sample_img5 = "/var/task/sample_5.jpg"
    sample_list=[sample_img1,sample_img2,sample_img3,sample_img4,sample_img5]
    for i in sample_list:
        pred=imageclassifier.run(i)
        prediction.append(pred)

    return {
        'prediction':prediction
    }
