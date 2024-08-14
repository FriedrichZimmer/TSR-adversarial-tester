"""
Copyright (c) 2024 Friedrich Zimmer
This script is starting the detection of alle three implemented detectors. Afterwards it creates a detection statistics.
"""
import os
from argparse import ArgumentParser

from detect_mass_fasterrcnn import classes_gtsrdb
from tools.statistics_for_complete_model import TsrStatistics
from util.detector_bestbox import BestBoxCropper
from util.detector_roboflow import RoboFlowDetModel
from util.detector_tensorflow import TFModel
from util.detector_yolo import YoloModel


def main(testproject_folder):
    threshold = 0.4  # detection with confidence below this threshold are ignored
    pixel_min = 30  # cropped images with pixel size below this value are ignored in classifier statistics

    detector_name = 'youssef_faster_rcnn'
    model_path = 'models/yousouf/saved_model'
    model = TFModel(model_path, threshold, detector_name, classes_gtsrdb)
    model.mass_detection(testproject_folder)

    detector_name = 'yashanksingh_train_1'
    model_path = 'models\\singh_detect\\best.pt'
    model = YoloModel(model_path, threshold, detector_name)
    model.mass_detection(testproject_folder)

    detector_name = 'smartcopilot'
    api_key = 'AiLy0Tt17OhmzTaP1TE9'
    model_id = 'smartcopilot/1'
    model = RoboFlowDetModel([model_id, api_key], threshold, detector_name)
    model.mass_detection(testproject_folder)

    best_box_cropper = BestBoxCropper('models/bestbox/box_results.csv', 0.0, 'Bestbox')
    best_box_cropper.mass_detection(args.testproject_folder)

    tsrstatistics = TsrStatistics(testproject_folder, None)
    tsrstatistics.analyse_tests(pixel_min)
    os.startfile(tsrstatistics.export_stat_excel())


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    main(args.testproject_folder)
