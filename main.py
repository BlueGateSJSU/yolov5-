import torch
import sys
import os

# Model
# model = torch.hub.load("../yolov5", 'custom',"yolov5s",source="local")  # or yolov5n - yolov5x6, custom

# Images
# img = "./data/images/dog_sample.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
os.system("python detect.py --weights runs/train/exp23/weights/best.pt --device cpu --save-txt --save-crop --source output/images/trainee2307")

# Training
# os.system("python train.py --img 640 --batch 8 --epochs 50 --data custom.yaml --weights yolov5s.pt --device cpu")

# Results
# results.crop()
# results.show()# or .show(), .save(), .crop(), .pandas(), etc.