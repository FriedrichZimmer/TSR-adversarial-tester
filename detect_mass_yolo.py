"""start detection with the detection model from Singh"""

from argparse import ArgumentParser

from util.detector_yolo import YoloModel

detector_name = 'yashanksingh_train_1'
threshold = 0.4
model_path = 'models\\singh_detect\\best.pt'
export_boxed = True


def main(testproject_folder):
    model = YoloModel(model_path, threshold, detector_name, export_boxed)
    model.mass_detection(testproject_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    main(args.testproject_folder)
