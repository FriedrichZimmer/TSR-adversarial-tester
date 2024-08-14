"""Copyright 2024 Friedrich Zimmer
This tool creates cropped images based on the images in the testproject_folder and the detection boxes from a
example file: models\bestbox\box_results.csv
"""

from argparse import ArgumentParser

from util.detector_bestbox import BestBoxCropper


def bestbox_cropping(testproject, boxfile):
    best_box_cropper = BestBoxCropper(boxfile, 0.0, 'Bestbox')
    best_box_cropper.mass_detection(testproject)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Path of the testproject')
    parser.add_argument('boxfile', type=str, help='Filepath of a csv file with detector results')
    args = parser.parse_args()
    bestbox_cropping(args.testproject_folder, args.boxfile)
