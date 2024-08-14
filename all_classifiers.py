"""
Copyright (c) 2024 Friedrich Zimmer
This script is starting the classification for all the cropped files within all detection folders of the project.
This script can only provide results if a detection model has been used on the testproject folder before.
"""
import os
from argparse import ArgumentParser

from classify_mass_keras_deepranjang import classes_gtsrdb
from classify_mass_roboflow_again import classes_again
from tools.statistics_for_complete_model import TsrStatistics

from util.classifier_keras import KerasClassModel
from util.classifier_roboflow import RoboFlowClassModel

CLASS_NOT_IN_CLASSIFIER = [
    ['No_Parking', 'deepranjang'],
    ['No_Parking', 'yashanksingh_v5'],
    ['KFC', 'deepranjang'],
    ['KFC', 'yashanksingh_v5'],
    ['KFC', 'again-3ibij'],
    ['Texaco', 'deepranjang'],
    ['Texaco', 'yashanksingh_v5'],
    ['Texaco', 'again-3ibij'],
    ['Speed_40', 'deepranjang'],
    ['Speed_40', 'yashanksingh_v5'],
    ['Speed_40', 'again-3ibij']
]


def main(testproject_folder):
    threshold = 0.6  # below this threshold a classification counts as undecided
    pixel_min = 30  # cropped images with pixel size below this value are ignored in classifier statistics
    # list of classes that are not known by certain classifiers

    classifier_name = 'yashanksingh_v5'
    model_path = 'models\\singh\\traffic_sign_classifier_v5.0_e10_b32.h5'
    IMG_SIZE_PP = 32
    equalizer = True
    model = KerasClassModel(model_path, classes_gtsrdb, threshold=threshold,
                            name=classifier_name, img_size=IMG_SIZE_PP, eq=equalizer)
    model.mass_classification(testproject_folder)
    classifier_name = 'deepranjang'
    model_path = 'models\\deepranjang\\Traffic.h5'
    IMG_SIZE_PP = 30  # image size for preprocessing
    equalizer = False
    model = KerasClassModel(model_path, classes_gtsrdb, threshold=threshold,
                            name=classifier_name, img_size=IMG_SIZE_PP, eq=equalizer)
    model.mass_classification(testproject_folder)
    classifier_name = 'again-3ibij'
    api_key = 'AiLy0Tt17OhmzTaP1TE9'
    model_id = 'fixed_dataset2.0/1'
    model = RoboFlowClassModel([model_id, api_key], classes_again, threshold, classifier_name)
    model.mass_classification(testproject_folder)

    tsrstatistics = TsrStatistics(testproject_folder, CLASS_NOT_IN_CLASSIFIER)
    tsrstatistics.analyse_tests(pixel_min)
    os.startfile(tsrstatistics.export_stat_excel())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    main(args.testproject_folder)
