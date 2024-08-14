"""
Copyright (c) 2024 Friedrich Zimmer
parent class for all detectors"""
import os
import csv
import cv2


def get_cropped_filename(result_folder, conf, image_name, boxid):

    return os.path.join(result_folder, 'cropped', f'{image_name[:-4]}_{boxid}_{round(int(conf * 100), 1)}.png')


class DetBoxFiles:
    """class for storing result data in csv"""

    def __init__(self, result_folder):
        # create the csv file with box data
        self.result_folder = result_folder
        # box_results is for storing the results in the format
        self.box_results_file = open(os.path.join(result_folder, 'box_results.csv'), 'w', newline='')
        self.box_results = csv.writer(self.box_results_file)
        # gt_detector is for using the found boxes as training data for a new detector model
        self.gt_detector_file = open(os.path.join(result_folder, 'gt_detector.csv'), 'w', newline='')
        self.gt_detector = csv.writer(self.gt_detector_file)
        # gt_classifier is for using the cropped images as training data for a new classificator model
        self.gt_classifier_file = open(os.path.join(result_folder, 'gt_classifier.csv'), 'w', newline='')
        self.gt_classifier = csv.writer(self.gt_classifier_file)

    def add_box(self, box, sign, orig_image, cropped_image):
        # box_results: original image, x1, y1, x2, y2, confidence category found by detector
        self.box_results.writerow([orig_image, box[0], box[1], box[2], box[3], box[4], box[5]])
        # box_results: original image, x1, y1, x2, y2, true category
        self.gt_detector.writerow([orig_image, box[0], box[1], box[2], box[3], sign])
        image_name = orig_image[orig_image.rfind('\\') + 1:]
        self.gt_classifier.writerow([cropped_image, sign, box[0], box[1], box[2], box[3]])

    def add_boxes(self, boxes, sign, orig_image):
        i = 0
        for box in boxes:
            self.add_box(box, sign, orig_image, "")
            i += 1

    def close_boxfile(self):
        self.box_results_file.close()
        self.gt_detector_file.close()
        self.gt_classifier_file.close()


class Detector:
    """superclass for the various detector classes includes the basic structure and folder creation"""

    def __init__(self, threshold=0.4, detector_name='Unnamed_Detector', export_boxed=False):
        """
        Initiates the detector

        Args:
            threshold (float): defines the threshold for the confidence. All Detections below this value get ignored
            detector_name (str): For labeling the directory for detection results
            export_boxed (bool): If true a copy of each image is created including the bounding boxes of the detection
        """
        self.threshold = threshold
        self.name = detector_name
        self.image_folder = None
        self.result_folder = None
        self.box_files = None
        self.sign = None
        self.export_boxed = export_boxed

    def get_image_list(self):
        """read paths of all image files into an array"""
        image_paths = []
        for image_file in os.listdir(self.image_folder):
            image_filepath = os.path.join(self.image_folder, image_file)
            if os.path.isfile(image_filepath) and (
                    image_file[len(image_file) - 3:] == 'jpg' or image_file[len(image_file) - 3:] == 'png'):
                print(f'Add file {image_filepath} to detection list.')
                image_paths.append(image_filepath)
        return image_paths

    def mass_detection(self, testproject_folder):
        """ cycle all signs in all tests """
        for self.sign in os.listdir(testproject_folder):
            sign_folder = os.path.join(testproject_folder, self.sign)
            if os.path.isfile(sign_folder):  # skip the statistics files stored in the main folder
                continue
            print('signfolder ' + sign_folder)
            for cam in os.listdir(sign_folder):
                if os.path.isfile(cam):  # there are not supposed to be any files in this folder
                    print('Warning: Files directly in sign folder detected')
                    continue
                self.image_folder = os.path.join(sign_folder, cam)
                print ('selfimage ' + self.image_folder)
                # create the result folder and cropped subfolder
                self.result_folder = os.path.join(self.image_folder, f'{self.name}_{self.threshold}')
                cropped_folder = os.path.join(self.result_folder, 'cropped')
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)

                self.box_files = DetBoxFiles(self.result_folder)

                self.single_detect()

    def single_detect(self):
        """ Will get overwritten by child classes"""
        pass


class DetectionResult:
    """class for storing and processing the results of a single image"""

    def __init__(self, image_path, result_folder):
        self.image_name = image_path[image_path.rfind('\\') + 1:]
        self.image = cv2.imread(image_path)
        self.boxes = []
        self.result_folder = result_folder

    def crop_images(self, box_files, sign):
        """Exports a cropped image for each detection of the image into the subfolder /cropped
            The coordinate format in boxes is xyxy
        """
        boxid = 0
        for box in self.boxes:
            cropped_filename = get_cropped_filename(self.result_folder, box[4], self.image_name, boxid)
            # cropping in opencv: first dimension row/y. second dimension columns/x
            cropped_image = self.image[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(cropped_filename, cropped_image)
            box_files.add_box(box, sign, self.image_name, cropped_filename)
            boxid += 1



    def create_boxed_image(self):
        """Exports a copy of the image with detection bounding boxes"""
        boxed_image = self.image

        for box in self.boxes:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            text_pos = (int(box[0]), int(box[1]) - 3)
            # rectangle in opencv: startpoint (x1,y1) endpoint (x2,y2)
            cv2.rectangle(boxed_image, p1, p2, (255, 0, 0), 3)
            # write confidence
            conf_text = f'{round(int(box[4] * 100), 1)} {box[5]}'
            cv2.putText(
                boxed_image,  # numpy array on which text is written
                conf_text,  # text
                text_pos,  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                1,  # font size
                (255, 0, 0, 255),  # font color
                2)
        boxed_filename = f'{self.result_folder}/boxed_{self.image_name}'
        # print(f'Saving boxed Image: {boxed_filename}')
        cv2.imwrite(boxed_filename, self.image)

    def remove_iou(self, threshold=0.3):
        """this method is needed if the detector model ignores intersect over union in case of different classifications

        Args:
            threshold (float): If intersect over union is above this value, the detection with the lower confidence
            gets deteled.
        """
        delete_boxes = []
        for box1 in self.boxes:
            for box2 in self.boxes:
                if box1 == box2:
                    continue
                if box1 in delete_boxes or box2 in delete_boxes:
                    continue
                # calculate area of overlap:
                # boxformat: xyxy
                # check if there is any overlay at all
                if box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1]:
                    continue
                aoo = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
                # print (f'Area of Overlay: {aoo}')
                # calculate areo of union:
                aou = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - aoo
                # print(f'Area of Union: {aou}')
                # calculate intersect over union:
                iou = aoo / aou
                # print(f'Intersect over Union: {iou}')
                # list box with the lower confidence for removal
                if iou > threshold:
                    print(f'IOU {iou} detected in {self.image_name}')
                    if box1[4] < box2[4]:
                        if box1 not in delete_boxes:
                            delete_boxes.append(box1)
                    else:
                        if box2 not in delete_boxes:
                            delete_boxes.append(box2)
        # remove boxes
        if len(delete_boxes) > 0:
            print(f'Removing {len(delete_boxes)} Boxes')
            # remove boxes
            for box in delete_boxes:
                if box in self.boxes:
                    print(f'Removing Box {box} because of Intersection')
                    self.boxes.remove(box)
