"""
Copyright (c) 2024 Friedrich Zimmer

class for classifiers based on Keras/Tensorflow"""

import os
from skimage import transform, exposure, io
import numpy as np
from time import time
from keras import models
from util.classifier_class import Classifier, ClassificationResult


class KerasClassModel(Classifier):

    def __init__(self, mp, class_list, threshold=0.6, name="Keras_Classifier", img_size=None, save_pp=False,
                 eq=False):
        super().__init__(threshold=threshold, name=name)
        self.class_list = class_list
        print(f'Loading Keras model {name} from {mp} as classifier')
        start_time = time()
        self.model = models.load_model(mp)
        end_time = time()
        print('Classifier Model loaded! Took {} seconds'.format(end_time - start_time))
        self.preprocessed_folder = None
        self.save_pp = save_pp
        self.eq = eq
        self.img_size = img_size

    def single_classification(self, image_path):
        """
        classifies all images in a single folder
        Args:
            image_path (str): path with cropped images

        Returns (ClassificationResult): Object with classification results
        """
        if self.save_pp:
            self.preprocessed_folder = os.path.join(self.classifier_folder, "_pp")
            if not os.path.exists(self.preprocessed_folder):
                os.makedirs(self.preprocessed_folder)

        loaded_image = io.imread(image_path)
        # Resizing images to the dimensions needed by the classifier model
        image = transform.resize(loaded_image, (self.img_size, self.img_size))
        # Applying Histogram Equalization to standardize lighting:
        if self.eq:
            image = exposure.equalize_adapthist(image, clip_limit=0.1)
        if self.save_pp:
            preprocessed_image = os.path.join(self.preprocessed_folder, image_path[image_path.rfind('\\') + 1:])
            print(f'saving preprocessed image as {preprocessed_image}')
            save_image = image * 255.0
            save_image = save_image.astype(np.uint8)
            io.imsave(f'preprocessed_image', save_image)
        if self.eq:
            image = image.astype('float32') / 255.0  # Normalizing image values between 0 and 1.

        image = image.reshape(1, self.img_size, self.img_size, 3)

        predictions = self.model.predict(image, verbose=0)

        classification_result = ClassificationResult(image_path)

        classification_result.found_class = self.class_list[np.argmax(predictions)]
        classification_result.confidence = np.amax(predictions)

        return classification_result
