"""
Copyright (c) 2024 Friedrich Zimmer
Detector class for Roboflow based models
"""

import time
from inference import get_model
from util.detector_class import Detector, DetectionResult


class RoboFlowDetModel(Detector):

    def __init__(self, model, threshold, detector_name, export_boxed=False):
        super().__init__(threshold, detector_name, export_boxed)

        print(f'Loading Roboflow model {detector_name}')
        start_time = time.time()
        self.model = get_model(model_id=model[0], api_key=model[1])
        end_time = time.time()
        print('Done! Took {} seconds'.format(end_time - start_time))

    def single_detect(self):
        """ run detection for all images in one folder and export the cropped images,
           the boxfile csv and optionally the complete image with detection boxes."""

        image_paths = self.get_image_list()
        if len(image_paths) == 0:
            print(f'No images in {self.image_folder}')
            return

        for image_path in image_paths:
            print(f'Detecting in {image_path}')
            result = self.model.infer(image=image_path, confidence=self.threshold, iou_threshold=0.5)
            detection_result = DetectionResult(image_path, self.result_folder)

            for r in result:
                for pred in r.predictions:
                    # box format : xyxy
                    # prediction format: center x, center y, width, height
                    box = [int(pred.x - 0.5 * pred.width), int(pred.y - 0.5 * pred.height),
                           int(pred.x + 0.5 * pred.width), int(pred.y + 0.5 * pred.height),
                           pred.confidence, pred.class_name]
                    detection_result.boxes.append(box)

            detection_result.remove_iou()
            detection_result.crop_images(self.box_files, self.sign)
            if self.export_boxed:
                detection_result.create_boxed_image()

