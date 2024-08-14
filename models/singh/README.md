# Classifier Model from Yashank Singh, Dev Singh and Gufran Ali

This Model is based on a Keras convolutional neural network and has been published in the research paper

"Yashank Singh, Dev Singh and Gufran Ali Traffic Sign Recognition using YOLOv8 Algorithm extended with CNN"

which can be found here:
https://github.com/yashanksingh/Traffic-Sign-Recognition/blob/master/assets/Traffic%20Sign%20Recognition%20Research%20Paper%20PBL%20v3.pdf

The pretrained model can be downloaded from Github:
https://github.com/yashanksingh/Traffic-Sign-Recognition

In our tests the pretrained model v5 was used, which can be downloaded here and then copied into this folder:
https://github.com/yashanksingh/Traffic-Sign-Recognition/raw/master/Models/traffic_sign_classifier_v5.0_e10_b32.model/saved_model.pb

For testing it, cropped images made with a detector or based on best box have to be available in the test folder.
For running the test execute:
python classify_mass_yashank_singh.py testfolder

The testfolder is usually safed in the format D:/results/20240801_1200_testfolder by the Carla TSR benchmark tool.
