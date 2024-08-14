# Detector Model from Yashank Singh, Dev Singh and Gufran Ali

This Model is based on YOLOv8 and has been published in the research paper

"Yashank Singh, Dev Singh and Gufran Ali Traffic Sign Recognition using YOLOv8 Algorithm extended with CNN"

which can be found here:
https://github.com/yashanksingh/Traffic-Sign-Recognition/blob/master/assets/Traffic%20Sign%20Recognition%20Research%20Paper%20PBL%20v3.pdf

The pretrained model can be downloaded from Github:
https://github.com/yashanksingh/Traffic-Sign-Recognition

In our tests the pretrained model train2 was used, which can be downloaded here and then copied into this folder:
https://github.com/yashanksingh/Traffic-Sign-Recognition/raw/master/runs/detect/train2/weights/best.pt

For testing execute the following python script:
python detect_mass_yolo.py testfolder

The testfolder is usually saved in the format D:/results/20240801_1200_testfolder by the Carla TSR benchmark tool.
