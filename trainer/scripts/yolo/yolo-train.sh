#!/bin/bash

# https://docs.ultralytics.com/modes/train/#usage-examples
yolo detect train data=data/deepfashion2.yaml model=yolov8n.pt epochs=100 imgsz=640 device=cpu workers=64
