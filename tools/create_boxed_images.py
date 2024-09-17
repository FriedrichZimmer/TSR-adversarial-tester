"""
Copyright (c) 2024 Friedrich Zimmer
create the boxed detector images from the boxex_info file information"""
import csv
import os
from argparse import ArgumentParser
from os import path, listdir, pardir
import cv2

from tools.statistics_for_complete_model import get_signs_from_name


def create_boxed_image(original_image, target_image, boxes, gt_cat=None):
    """create a boxinfo image based on an original image and the boxes data"""
    print(original_image)

    # get true and adversarial traffic sign:
    actual_sign, adv_sign = get_signs_from_name(gt_cat)

    if path.isfile(original_image) and (
            original_image[len(original_image) - 3:] == 'jpg' or original_image[len(original_image) - 3:] == 'png'):
        img = cv2.imread(original_image)
        for box in boxes:

            if gt_cat:
                if box[6] == 'Below threshold':
                    color = (255, 0, 0)  # Blue, if classifier confidence below threshold
                    text = f'{round(int(float(box[5]) * 100), 1)} {box[6]}'
                elif box[6] == actual_sign:
                    color = (0, 255, 0)  # Green, if classification is corret
                    text = f'{round(int(float(box[5]) * 100), 1)} {box[6]}'
                elif box[6] == adv_sign:
                    color = (0, 0, 255)  # Red, if adversarial attack is sucessfull
                    text = f'{round(int(float(box[5]) * 100), 1)} {box[6]}'
                else:  # other wrong detections are yellow
                    color = (0, 255, 255)  # Yellow for any other detection
                    text = f'{round(int(float(box[5]) * 100), 1)} {box[6]}'
            else:
                color = (255, 0, 0)  # Blue if no classification was done
                text = f'{round(int(float(box[5]) * 100), 1)}'  # text

            # box: orig_file, x1, y1, x2, y2, conf, detcat
            p1 = (int(box[1]), int(box[2]))
            p2 = (int(box[3]), int(box[4]))
            cv2.rectangle(img, p1, p2, color, 2)
            # read confidence from cropped filename
            # print(text)

            cv2.putText(
                img,  # numpy array on which text is written
                text,
                (int(box[1]), int(box[2]) - 4),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                0.6,  # font size
                color,  # font color green
                2)
        print(f'writing {target_image}')
        cv2.imwrite(target_image, img)


def get_rows_from_csv(filename):
    file = open(filename, 'r')
    reader = csv.reader(file)
    # read the content into an array variable
    rows = []
    for row in reader:
        if len(row) > 0:
            rows.append(row)
    return rows


def create_boxi_detector(det_path):
    """creates boxinfo images for all images in a certain detector folder"""
    boxinfo_path = path.join(det_path, 'box_results.csv')
    if not path.exists(boxinfo_path):
        print('ERROR: box_results.csv not found in detector directory!')
        exit(boxinfo_path)

    print(f'creating images from {boxinfo_path}')

    # opening csv file with box info
    rows = get_rows_from_csv(boxinfo_path)

    # boxinfo_file = open(boxinfo_path, 'r')
    # boxinfo_reader = csv.reader(boxinfo_file)
    # # read the content into an array variable
    # rows =[]
    # for row in boxinfo_reader:
    #     if len(row) > 0:
    #         rows.append(row)

    # get parent path that contains the original images
    original_image_path = path.abspath(path.join(det_path, pardir))
    print(original_image_path)

    for original_image in listdir(original_image_path):
        original_name = path.join(original_image_path, original_image)
        if not os.path.isfile(original_name):
            continue
        print(f'Reading {original_name}')
        boxes = []
        for row in rows:
            if len(row) > 0:
                # print(f'{original_name} {row[0]}')
                if original_name == row[0]:
                    boxes.append(row)

        print(f'found {len(boxes)} Traffic signs in this image')
        target_name = path.join(det_path, original_image)
        create_boxed_image(original_name, target_name, boxes)


def create_boxi_classificator(class_path, pixel_min=20, threshold=0.6):
    """creates boxed images with the classification results

    Args:
        class_path (str): Path to the classifier results
        pixel_min (int): Boxes with x or y size below this value are ignored
        threshold (float): box is blue if confidence is below threshold
    """
    cr_path = path.join(class_path, 'classifier_results.csv')
    if not path.exists(cr_path):
        print('ERROR: box_results.csv not found in detector directory!')
        exit(cr_path)

    print(f'creating images from {cr_path}')

    # opening csv file with classification results
    cr_rows = get_rows_from_csv(cr_path)

    # opening csv file with detection results:
    det_path = class_path[:class_path.find('cropped')]
    print(det_path)
    dr_path = path.join(det_path, 'gt_classifier.csv')
    dr_rows = get_rows_from_csv(dr_path)

    # combining result data from detection and classification
    combined_rows = []
    for cr_row in cr_rows:
        for dr_row in dr_rows:
            # cr_row: cropped_image, detectedsign, confidence
            # dr_row: cropped_image, truesign, x1,y1, x2,y2
            # skip small images
            if (int(dr_row[4])-int(dr_row[2]) < pixel_min) or (int(dr_row[5])-int(dr_row[3]) < pixel_min):
                continue
            if cr_row[0] == dr_row[0]:
                combined_rows.append([cr_row, dr_row])

    if len(combined_rows)<1:
        print(f'No traffic signs with size>={pixel_min} found.')

    # get parent path that contains the original images
    original_image_path = path.abspath(path.join(det_path, pardir))
    print(original_image_path)

    # cycle original images from carla
    for original_image in listdir(original_image_path):
        original_name = path.join(original_image_path, original_image)
        if not os.path.isfile(original_name):
            continue
        print(f'Reading {original_name}')
        boxes = []
        true_sign = combined_rows[0][1][1]
        # cycle all found boxes from csv
        for comb_row in combined_rows:
            # cr_row: cropped_image, detectedsign, confidence
            # dr_row: cropped_image, truesign, x1,y1, x2,y2
            # box: orig_file, x1, y1, x2, y2, conf, detcat

            backslash = '\\'
            image_nr_from_cropped = comb_row[0][0][comb_row[0][0].rfind(backslash)+1:comb_row[0][0].rfind(backslash)+5]
            # print(f'{original_image[:4]} {image_nr_from_cropped}')

            if original_image[:4] == image_nr_from_cropped:
                # check for threshold

                if float(comb_row[0][2])< threshold:
                    comb_row[0][1] = "Below threshold"

                boxes.append([original_name, comb_row[1][2], comb_row[1][3], comb_row[1][4], comb_row[1][5],
                              comb_row[0][2], comb_row[0][1]])

        print(f'Found {len(boxes)} Traffic signs in this image')
        target_name = path.join(class_path, original_image)
        create_boxed_image(original_name, target_name, boxes, true_sign)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', type=str, help='Filepath and name of the detector directory')
    parser.add_argument('-c', type=str, help='Filepath and name of the classifier directory')
    args = parser.parse_args()
    if args.d:
        create_boxi_detector(args.d)
    if args.c:
        create_boxi_classificator(args.c)
