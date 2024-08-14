# TSR adversarial tester

## Description
Testing tool that can test various CV Models for Traffic Sign recognition. It is testing both detectors and 
classifiers and requires a simulation result folder from the CARLA TSR Adversarial benchmark. 

https://github.com/FriedrichZimmer/CARLA-TSR-adversarial-benchmark

This analysis results in a statistics csv file. Creation of videos with detection and classification results as boxes is also supported.

## Installation
1. This was tested with Python 3.10 on Windows 11. Install the needed python frameworks with pip. Please keep in mind that different models might need different versions of the same framework.
```
pip install numpy, opencv-python, Pillow, skimage, inference, keras, tensorflow, ultralytics, pandas, openpyxl
```
2. Copy the models you want to test onto the hard drive. The links to the models used as example can be found in the text files in the models section. Roboflow models don't require a downloaded model but just an active internet connection. 

## Demo

For the demo the modules skimage, keras, tensorflow and ultralytics are not needed.
```
python demo.py
```
This demo will detect traffic signs in the images in the demo folder, classify them and create a statistic and two videos that show the bounding boxes and classifications.

Result of traffic sign detection and classification with a normal Speed 100 Sign:
![Speed_100](/demo/demo_image.png)
Result of traffic sign detection and classification with an adversarial Speed 100#120 Sign:
![Speed_100#120_Adversarial](/demo/demo_image_adv.png)

## Usage
For doing those things step by step use the following commands:


1. Go to the project folder and start detections with the demo folder included in this project. For testing detector models based on tensorflow or yolo, they first need to be downloaded. Details and locations of tested models can be found in the model folders.
```
python all_detectors.py demo
```
If you only want to use a single detector you can also use those commands:
```
python detect_mass_fasterrcnn.py demo
python detect_mass_roboflow_again.py demo
python detect_mass_yashank_singh.py demo
```

2. Create cropped images based on box coordinates stored in a csv file. Those are made, so during classification, identically cropped images are used when comparing different weather or camera. 
```
python bestbox_cropping.py demo models/bestbox/box_results.csv
```
3. Start classifications based on the cropped images. For testing classifier models based on Keras, they first need to be downloaded. Details and locations of tested models can be found in the model folders.

```
python all_classifiers.py demo
```
If you only want to use a single classifier you can also use those commands:
```
python classify_mass_keras_deepranjang.py demo
python classify_mass_roboflow_again.py demo
python classify_mass_yashank_singh.py demo
```
4. Create a statistics of the result. This will create a csv file and open it.
```
python tools/statistics_for_complete_model.py demo
```
5. Create images with detection and classification boxes. For this, the path to a classifier results has to be given. 
```
python tools/create_boxed_images.py -c demo/Speed_100#Speed_120/01_default_new/Bestbox_0.0/cropped/again-3ibij_0.6
```
This will create images based on the original images with bounding boxes showing the classification result. The first value is the confidence of the classification result and the second one is the detected category.
THe colors show the success of the the classification (green=correct, blue=confidence below threshold, yellow=false positive, red=adversarial sucess)

It is also possible to create such images just with detection results:
```
python tools/create_boxed_images.py -d demo/Speed_100#Speed_120/01_default_new/Bestbox_0.0
```

6. Create a mp4 video file with these boxed images. avi is also supported.
```
python tools/convert_img_video.py demo/Speed_100#Speed_120/01_default_new/Bestbox_0.0/cropped/again-3ibij_0.6
```

## Roadmap
Until my Master Thesis is finished end of August 2024, I will continuously update this project. After this, I probably will abandon this project.

Some Change ideas:
-[ ] Make statistic based on csv instead of the number of images. This will make calculation much faster.
-[ ] Add support for other machine learning frameworks and model formats.
-[ ] Add a ground truth for the detectors
-[ ] Train traffic sign detection and classification models based the original and cropped images. As all images are having traffic signs in the exact same location and the same light conditions an evenly distributed training data.

## Support
You can contact me at friedrich.zimmer@arrk-engineering.com.

## License
This project is licensed under the MIT license.
Copyright 2024 (c) Friedrich Zimmer at ARRK Engineering

## Project status


***