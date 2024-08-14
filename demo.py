"""
Copyright (c) 2024 Friedrich Zimmer
This script demonstrates all features of this tool based on the images in the demo folder and the CV-models from Roboflow
"""
import os

from bestbox_cropping import bestbox_cropping
from classify_mass_roboflow_again import classify_mass_roboflow
from detect_mass_roboflow import detect_mass_roboflow
from tools.convert_img_video import gen_video
from tools.create_boxed_images import create_boxi_classificator
from tools.statistics_for_complete_model import simple_statistics

# start roboflow detection with the demo folder included in this project.
detect_mass_roboflow("demo")
# Create cropped images based on box coordinates stored in a csv file
bestbox_cropping("demo", "models/bestbox/box_results.csv")
# Start roboflow classification based on the cropped images from detector and csv_cropper
classify_mass_roboflow("demo")
# Create a statistics of the result
simple_statistics("demo")
# Create images with detection and classification boxes for both normal and adversarial traffic signs
create_boxi_classificator("demo/Speed_100/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6")
create_boxi_classificator("demo/Speed_100#Speed_120/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6")
# Create videos with these images.
gen_video("demo/Speed_100/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6")
gen_video("demo/Speed_100#Speed_120/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6")
# Watch and compare the created video
os.startfile(os.path.join(os.path.abspath(os.getcwd()),
                          "demo/Speed_100/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6/_videos/video"
                          ".mp4"))
os.startfile(os.path.join(os.path.abspath(os.getcwd()),
                          "demo/Speed_100#Speed_120/10_mblur_low/Bestbox_0.0/cropped/again-3ibij_0.6/_videos/video"
                          ".mp4"))
