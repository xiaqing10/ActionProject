#!/bin/bash

BASEDIR=$(dirname "$0")
DIR=$BASEDIR/../fastmot/models

set -e

pip3 install gdown

gdown https://drive.google.com/uc?id=1-kXZpA6y8pNbDMMD7N--IWIjwqqnAIGZ -O $DIR/yolov4_crowdhuman.onnx
