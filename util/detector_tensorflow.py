"""
Copyright (c) 2024 Friedrich Zimmer
Object Detection based on tensorflow
"""
import tensorflow as tf
import time
from PIL import Image
import numpy as np
from util.detector_class import Detector, DetectionResult


class TFModel(Detector):

    def __init__(self, model_path, threshold, detector_name, class_list, export_boxed=False):
        super().__init__(threshold, detector_name, export_boxed)

        # Enable GPU dynamic memory allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.class_list = class_list

        print(f'Loading Tensorflow model {model_path}')
        start_time = time.time()
        self.model = tf.saved_model.load(model_path)
        end_time = time.time()
        print('Done! Took {} seconds'.format(end_time - start_time))

    def single_detect(self):
        """
        detects traffic signs in all street scenario images in a single folder
        """

        image_paths = self.get_image_list()
        if len(image_paths) == 0:
            print(f'No images in {self.image_folder}')
            return

        for image in image_paths:
            print(f'Detecting in {image}')

            loaded_image = Image.open(image)
            loaded_image.convert("RGB")
            image_np = np.array(loaded_image.convert("RGB"))
            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(image_np)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.model(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.

            num_detections = int(detections.pop('num_detections'))
            # print("num detections: " + str(num_detections))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            detection_result = DetectionResult(image, self.result_folder)

            image_height, image_width, channels = image_np.shape

            for index, score in enumerate(detections['detection_scores']):
                if score < self.threshold:
                    continue

                label = self.class_list[detections['detection_classes'][index] - 1]
                ymin, xmin, ymax, xmax = detections['detection_boxes'][index]

                detection_result.boxes.append(
                    [int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                     int(ymax * image_height), score, label])

            detection_result.remove_iou()
            detection_result.crop_images(self.box_files, self.sign)
            if self.export_boxed:
                detection_result.create_boxed_image()
            # self.box_files.add_boxes(detection_result.boxes, self.sign, image)
