import cv2
import numpy as np
from keras.models import load_model


class LogoClassifier:
    def __init__(self, model_path, labels_path):
        try:
            self.model = load_model(model_path, compile=False)
            self.class_names = open(labels_path, "r").readlines()
            print("Logo classifier loaded successfully")
        except Exception as e:
            print(f"Error loading logo classifier: {e}")
            self.model = None
            self.class_names = []

    def predict(self, image):
        if self.model is None:
            return "Model not loaded", 0.0

        try:
            # Ensure image is in correct format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # Preprocess image for the model
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_normalized = (image_array / 127.5) - 1

            # Make prediction
            prediction = self.model.predict(image_normalized, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index].strip() if self.class_names else "Unknown"
            confidence_score = float(prediction[0][index])

            return class_name, confidence_score

        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0