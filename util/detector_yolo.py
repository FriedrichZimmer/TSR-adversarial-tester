"""
Copyright (c) 2024 Friedrich Zimmer
class for Detectors based on Yolo
"""
import os
import time
from ultralytics import YOLO
from util.detector_class import Detector, DetectionResult


class YoloModel(Detector):

    def __init__(self, model_path, threshold, detector_name, export_boxed=False):
        super().__init__(threshold, detector_name, export_boxed)

        print(f'Loading Yolo model {model_path}')
        start_time = time.time()
        self.model = YOLO(model_path)
        end_time = time.time()
        print('Done! Took {} seconds'.format(end_time - start_time))

        # Display model information (optional)
        self.model.info()
        print(self.model.names)

    def single_detect(self):
        """ run detection for all images in one folder and export the cropped images,
           the boxfile csv and optinally the compete image with detection boxes."""
        results = self.model(self.image_folder, stream=True, conf=self.threshold)

        for result in results:
            # get path of original image
            image_name = result.path[result.path.rfind('\\') + 1:]
            detection_result = DetectionResult(result.path, self.result_folder)
            # saving original images with detection boxes
            if self.export_boxed:
                save_file = os.path.join(self.result_folder, f'boxed_{image_name}')
                print(f'Export boxed image as {save_file}')
                result.save(filename=save_file)

            boxes = result.boxes
            if boxes is not None:
                # Boxes object for bounding box outputs
                for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), conf, cls]
                    detection_result.boxes.append(box)

            detection_result.crop_images(self.box_files, self.sign)

            if self.export_boxed:
                detection_result.create_boxed_image()