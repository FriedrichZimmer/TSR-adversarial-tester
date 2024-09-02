"""
Copyright (c) 2024 Friedrich Zimmer
class for classifiers from Roboflow"""

from time import time

from util.classifier_class import Classifier, ClassificationResult
from inference import get_model


class RoboFlowClassModel(Classifier):
    def __init__(self, model, class_list, threshold=0.6, name='Roboflow_Classifier'):
        """
            Class for classifieng traffic signs with Keras/Tensorflow based detectors
            Args:
            model (str,str): model ID and api key of the used model
            class_list [str]: Array of string values with the category names sorted by category ID of the model
            threshold (float): threshold for filtering out unreliable results
            name (str): Name of the classifier model
        """
        super().__init__(threshold=threshold, name=name)
        self.category_list = class_list
        print(f'Loading Roboflow model {name} as classifier')
        start_time = time()
        self.model = get_model(model_id=model[0], api_key=model[1])
        end_time = time()
        print('Classifier Model loaded! Took {} seconds'.format(end_time - start_time))

    def single_classification(self, image_path):
        """
        classifies all images in a single folder
        Args:
            image_path (str): path with cropped images

        Returns (ClassificationResult): Object with classification results
        """

        # confidence threshold for the first detection. for getting confidence value, even if it's below the threshold
        result = self.model.infer(image=image_path, confidence=0.3)

        classification_result = ClassificationResult(image_path)

        for r in result:
            if len(r.predicted_classes) == 1:
                print(self.category_list[int(r.predicted_classes[0])])
                classification_result.found_class = self.category_list[int(r.predicted_classes[0])]
                print(r.predictions[r.predicted_classes[0]].confidence)
                classification_result.confidence = r.predictions[r.predicted_classes[0]].confidence
            # if 0 or more than one classes are detected, the result isn't distinct, so it's counted as below threshold

        return classification_result
