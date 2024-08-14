# DeepranjanG Detector Model

This has been published on GitHub

https://github.com/DeepranjanG/Traffic_sign_classification

To test the pretrained model, the file Traffic.h5 has to be downloaded and copied into the folder /models/deepranjang

https://github.com/DeepranjanG/Traffic_sign_classification/blob/main/Traffic.h5

For testing it, cropped images made with a detector or based on best box have to be available in the test folder.
For running the test execute:
python classify_mass_keras_deepranjang.py testfolder

The testfolder is usually safed in the format D:/results/20240801_1200_testfolder by the Carla TSR benchmark tool.
