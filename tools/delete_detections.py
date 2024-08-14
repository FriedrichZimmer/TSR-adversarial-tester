"""Copyright (c) 2024 Friedrich Zimmer
Delete all detection folders.
(If you want to restart the analysis from the start)"""

import os
import shutil
from argparse import ArgumentParser


def del_all_detection_results(result_folder):
    """ delete all detection folders in this project
    """
    for sign in os.listdir(result_folder):
        sign_folder = os.path.join(result_folder, sign)
        if os.path.isfile(sign_folder):  # skip the statistics files stored in the main folder
            continue
        for cam in os.listdir(sign_folder):
            # define the folders used for the results
            cam_folder = os.path.join(sign_folder, cam)
            for detector in os.listdir(cam_folder):
                det_path = os.path.join(cam_folder, detector)
                if not os.path.isfile(det_path):
                    print(f'Deleting {det_path}')
                    shutil.rmtree(det_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    del_all_detection_results(args.testproject_folder)
