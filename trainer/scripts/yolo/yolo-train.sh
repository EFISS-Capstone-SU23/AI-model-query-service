#!/bin/bash

# https://docs.ultralytics.com/modes/train/#usage-examples
yolo detect train data=data/deepfashion2.yaml model=yolov8n.pt epochs=12 imgsz=640 device=0 workers=64 batch=-1
