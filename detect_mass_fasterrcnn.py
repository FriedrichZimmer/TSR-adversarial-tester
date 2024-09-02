"""This script is starting detection with the model of youssef for all images in a test folder"""

from argparse import ArgumentParser

from util.detector_tensorflow import TFModel

detector_name = 'youssef_faster_rcnn'
threshold = 0.4
model_path = 'models/yousouf/saved_model'
export_boxed = True

classes_gtsrdb = ['Speed_20',
                  'Speed_30',
                  'Speed_50',
                  'Speed_60',
                  'Speed_70',
                  'Speed_80',
                  'Speed_80_End',
                  'Speed_100',
                  'Speed_120',
                  'No_Over',
                  'No_Over_Heavy',
                  'Right-of-Way_next',
                  'Priority_Road',
                  'Yield',
                  'Stop',
                  'No_Vehicles',
                  'Heavy_Veh_Proh',
                  'No_Entry',
                  'General_Caution',
                  'Dang_Left_C',
                  'Dang_Right_C',
                  'Double_Curve',
                  'Bumpy_Road',
                  'Slippery_Road',
                  'Narrowing_Road',
                  'Road_Work',
                  'Traffic_Sig',
                  'Pedestrian',
                  'Children',
                  'Bike',
                  'Snow',
                  'Deer',
                  'End_of_Limits',
                  'Turn_Right_Ahead',
                  'Turn_Left_Ahead',
                  'Ahead_Only',
                  'Go_Straight_Right',
                  'Go_Straight_Left',
                  'Keep_Right',
                  'Keep_Left',
                  'Roundabout_Mandatory',
                  'End_No_Over',
                  'End_No_Over_Heavy']

def main(testprojec_folder):
    model = TFModel(model_path, threshold, detector_name, classes_gtsrdb, export_boxed)
    model.mass_detection(testprojec_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    main(args.testproject_folder)

