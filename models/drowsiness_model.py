import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class DrowsinessModel:
    def __init__(self, model_path="ds_project.h5"):
        self.model = load_model(model_path)

    def predict_eye_state(self, eye_image):
        """Predicts if the eye is open or closed."""
        eye_image = eye_image.astype('float') / 255.0
        eye_image = img_to_array(eye_image)
        eye_image = np.expand_dims(eye_image, axis=0)
        pred = self.model.predict(eye_image)
        return np.argmax(pred)
