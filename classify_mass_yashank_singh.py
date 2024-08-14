"""This example script is evaluating the classification model from Yashank Singh
This script can only provide results if a detection model has been used on the testproject folder before.
"""
from argparse import ArgumentParser

from util.classifier_keras import KerasClassModel

classifier_name = 'yashanksingh_v5'
model_path = 'models\\singh\\traffic_sign_classifier_v5.0_e10_b32.h5'
IMG_SIZE_PP = 32
used_threshold = 0.6
equalizer = True

# the classifier of Yashank Singh was trained on the GTSRDB Dataset. Therefore those 43 categories are used.
categories_gtsrdb = ['Speed_20',
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


def main(testproject_folder):
    model = KerasClassModel(model_path, categories_gtsrdb, threshold=used_threshold,
                            name=classifier_name, img_size=IMG_SIZE_PP, eq=equalizer)
    model.mass_classification(testproject_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    main(args.testproject_folder)
