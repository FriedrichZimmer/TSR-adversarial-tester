# Model from: https://universe.roboflow.com/emposo/smartcopilot/model/1
#  @misc{
#     smartcopilot_dataset,
#     title = { SmartCoPilot Dataset },
#     type = { Open Source Dataset },
#     author = { Emposo },
#     howpublished = { \url{ https://universe.roboflow.com/emposo/smartcopilot } },
#     url = { https://universe.roboflow.com/emposo/smartcopilot },
#     journal = { Roboflow Universe },
#     publisher = { Roboflow },
#     year = { 2023 },
#     month = { nov },
#     note = { visited on 2024-06-18 },
#     }

from argparse import ArgumentParser

from util.detector_roboflow import RoboFlowDetModel

detector_name = 'smartcopilot'
threshold = 0.4
api_key = 'AiLy0Tt17OhmzTaP1TE9'
model_id = 'smartcopilot/1'
export_boxed = True


def detect_mass_roboflow(testproject_folder):
    model = RoboFlowDetModel([model_id, api_key], threshold, detector_name, export_boxed)
    model.mass_detection(testproject_folder)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('testproject_folder', type=str, help='Filepath and name of the results directory')
    args = parser.parse_args()
    detect_mass_roboflow(args.testproject_folder)
