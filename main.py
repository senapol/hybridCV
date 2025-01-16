from ultralytics import YOLO
import cv2
import torch
import os

# YOLOv6n is fastest but least accurate model
# mAP accuracy can be increased by training
model = YOLO('yolov8n.pt')

if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs=50, patience=5)