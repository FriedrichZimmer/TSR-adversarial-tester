"""start classification with the model from again for all images in the testfolder
This script can only provide results if a detection model has been used on the testproject folder before."""
# Model Source:
# https://universe.roboflow.com/again-3ibij/fixed_dataset2.0
#  fixed_dataset2.0_dataset,
#  title = { fixed_dataset2.0 Dataset },
#  type = { Open Source Dataset },
#  author = { again },
#  howpublished = { \url{ https://universe.roboflow.com/again-3ibij/fixed_dataset2.0 } },
#  url = { https://universe.roboflow.com/again-3ibij/fixed_dataset2.0 },
#  journal = { Roboflow Universe },
#  publisher = { Roboflow },
#  year = { 2023 },
#  month = { dec },
#  note = { visited on 2024-04-23 },

from argparse import ArgumentParser

from util.classifier_roboflow import RoboFlowClassModel

classifier_name = 'again-3ibij'
threshold = 0.6
api_key = 'AiLy0Tt17OhmzTaP1TE9'
model_id = 'fixed_dataset2.0/1'

# this classifier uses an extended version of GTSRDS
classes_again = ['Speed_20',
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
                 'Double_Curve_left',
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
                 'End_No_Over_Heavy',
                 'No_Halting',
                 'corner',
                 'Keep_Right_Left',
                 'Parking',
                 'No_Parking',
                 'No_Turn_right',
                 'Double_curve_right',
                 'No_Turn_left',
                 'U-Turn'
                 ]


def classify_mass_roboflow(testproject_folder):
    model = RoboFlowClassModel([model_id, api_key], classes_again, threshold, classifier_name)
    model.mass_classification(testproject_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    classify_mass_roboflow(args.testproject_folder)
