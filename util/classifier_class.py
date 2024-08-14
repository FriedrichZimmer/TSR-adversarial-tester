"""
Copyright (c) 2024 Friedrich Zimmer
parent class for all classifiers"""
import csv
import os
import shutil


class Classifier:
    """superclass for the various detector classes includes the basic structure and folder creation"""

    def __init__(self, threshold=0.6, name='Unnamed_Classifier'):
        """
        Initiates the classifier basic values

        Args:
            threshold (float): confidence threshold for a sucessful classification
            name (str): for labeling the result folder
        """
        self.threshold = threshold
        self.name = name
        self.result_folder = None
        self.result_csv = None
        self.sign = None
        self.classifier_folder = None

    def mass_classification(self, testproject_folder):
        """ cycle all detector_results in all signs in all tests """
        for test in os.listdir(testproject_folder):
            test_folder = os.path.join(testproject_folder, test)
            if os.path.isfile(test_folder):
                continue
            for cam in os.listdir(test_folder):
                cam_folder = os.path.join(test_folder, cam)
                for detector in os.listdir(cam_folder):
                    detector_folder = os.path.join(cam_folder, detector)
                    if os.path.isfile(detector_folder) or detector == '_videos':
                        continue
                    cropped_folder = f'{detector_folder}\\cropped'
                    print(self.name + ': Analyzing cropped images in ' + cropped_folder)

                    self.classifier_folder = os.path.join(cropped_folder, f'{self.name}_{self.threshold}')
                    # create classifier folder
                    if not os.path.exists(self.classifier_folder):
                        os.makedirs(self.classifier_folder)

                    # record results in a csv file
                    c_results_file = open(os.path.join(self.classifier_folder, 'classifier_results.csv'), 'w', newline='')
                    c_results = csv.writer(c_results_file)

                    for image_name in os.listdir(cropped_folder):
                        image_path = os.path.join(cropped_folder, image_name)
                        if not os.path.isfile(image_path) or image_path[len(image_path) - 3:] != 'png':
                            continue

                        print(f'Classifying {image_path}')
                        result = self.single_classification(image_path)

                        c_results.writerow([image_path, result.found_class, result.confidence])

                        if result.confidence > self.threshold:
                            SORTED_FOLDER = os.path.join(self.classifier_folder, result.found_class)
                        else:
                            SORTED_FOLDER = os.path.join(self.classifier_folder, '_below_threshold')

                        if not os.path.exists(SORTED_FOLDER):
                            os.makedirs(SORTED_FOLDER)

                        filename_conf = f'{SORTED_FOLDER}\\{round(result.confidence * 100, 1)}_{image_name}'

                        shutil.copy(image_path, filename_conf)

                    c_results_file.close()

    def single_classification(self, image_path):
        return ClassificationResult(image_path)


class ClassificationResult:
    """class for storing and processing the results of a single image"""

    def __init__(self, image_path):
        self.image_name = image_path[image_path.rfind('\\') + 1:]
        self.confidence = 0.0
        self.found_class = '_below_threshold'
