"""
Copyright (c) 2024 Friedrich Zimmer
Class for cropping images based on certain previous detection results or a predefined ground truth
"""

import time


from tools.create_boxed_images import get_rows_from_csv
from util.detector_class import Detector, DetectionResult


class BestBoxCropper(Detector):
    """Class for cropping images based on certain previous detection results.
    It's based on the detector in order to use the same cropping functions """

    def __init__(self, boxfile, threshold, detector_name="Best_Box", export_boxed=False):
        """ Initiate cropping
        Args:
            boxfile (str): path to the csv file that contains box data for all frames
            threshold (float): not relevant for this class
            detector_name (str): Name used for the folder that contains the cropped files.
            export_boxed (bool): Choose true if creation of images with detection boxes is desired.
        """
        super().__init__(threshold, detector_name, export_boxed)

        print(f'Loading Roboflow model {detector_name}')
        start_time = time.time()
        # read cropping coordinates
        self.crop_coords = get_rows_from_csv(boxfile)
        end_time = time.time()
        print('Done! Took {} seconds'.format(end_time - start_time))

    def single_detect(self):
        """ crop images for all images in one folder based on best_box,
           and optionally the complete image with detection boxes."""

        image_paths = self.get_image_list()
        if len(image_paths) == 0:
            print(f'No images in {self.image_folder}')
            return

        for image_path in image_paths:

            detection_result = DetectionResult(image_path, self.result_folder)

            for bestbox in self.crop_coords:
                # print(f'{bestbox[0][-8:]} {image_path[-8:]}')
                if bestbox[0] == image_path[-8:]:
                    # box format : xyxy, confidence and detected category are not relevant.
                    box = [int(bestbox[1]), int(bestbox[2]), int(bestbox[3]), int(bestbox[4]), 1.0, 'bestbox']
                    print(f'{image_path[-8:]} will be cropped with {box}')
                    detection_result.boxes.append(box)

            detection_result.remove_iou()
            detection_result.crop_images(self.box_files, self.sign)
            if self.export_boxed:
                detection_result.create_boxed_image()

