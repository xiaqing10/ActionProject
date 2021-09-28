import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import trt_pose.coco
import math
import os
import numpy as np
import traitlets

with open('trt_pose_hand/preprocess/hand_pose.json', 'r') as f:
    hand_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(hand_pose)
import trt_pose.models

num_parts = len(hand_pose['keypoints'])
num_links = len(hand_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
import torch


WIDTH = 224
HEIGHT = 224
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


class InputReNormalization(torch.nn.Module):
    """
        This defines "(input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]" custom operation
        to conform to "Unit" normalized input RGB data.
    """
    def __init__(self):
        super(InputReNormalization, self).__init__()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).reshape((1,3,1,1)).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).reshape((1,3,1,1)).cuda()

    def forward(self, x):
        return (x - self.mean) / self.std



class HeatmapMaxpoolAndPermute(torch.nn.Module):
    """
        This defines MaxPool2d(kernel_size = 3, stride = 1) and permute([0,2,3,1]) custom operation
        to conform to [part_affinity_fields, heatmap, maxpool_heatmap] output format.
    """
    def __init__(self):
        super(HeatmapMaxpoolAndPermute, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        heatmap, part_affinity_fields = x
        maxpool_heatmap = self.maxpool(heatmap)

        part_affinity_fields = part_affinity_fields.permute([0,2,3,1])
        heatmap = heatmap.permute([0,2,3,1])
        maxpool_heatmap = maxpool_heatmap.permute([0,2,3,1])
        return [part_affinity_fields, heatmap, maxpool_heatmap]


class HeatmapMaxpoolAndPermute(torch.nn.Module):
    """
        This defines MaxPool2d(kernel_size = 3, stride = 1) and permute([0,2,3,1]) custom operation
        to conform to [part_affinity_fields, heatmap, maxpool_heatmap] output format.
    """
    def __init__(self):
        super(HeatmapMaxpoolAndPermute, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        heatmap, part_affinity_fields = x
        maxpool_heatmap = self.maxpool(heatmap)

        part_affinity_fields = part_affinity_fields.permute([0,2,3,1])
        heatmap = heatmap.permute([0,2,3,1])
        maxpool_heatmap = maxpool_heatmap.permute([0,2,3,1])
        return [part_affinity_fields, heatmap, maxpool_heatmap]


MODEL_WEIGHTS = 'trt_pose_hand/model/hand_pose_resnet18_att_244_244.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

# import torch2trt
# model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<30)
# OPTIMIZED_MODEL = 'hand_trt.pth'
# torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


# converted_model = torch.nn.Sequential(InputReNormalization(), model, HeatmapMaxpoolAndPermute())

# Define input and output names for ONNX exported model.
input_names = ["input"]
output_names = ["cmap","paf"]

# Export the model to ONNX.
dummy_input = torch.zeros((1, 3, 224,224)).cuda()
torch.onnx.export(model, dummy_input, "hand.onnx",
                    input_names=input_names, output_names=output_names)

os.system("trtexec --onnx=hand.onnx --saveEngine=./trt_hand.engine")
