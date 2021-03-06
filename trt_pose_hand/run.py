import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import trt_pose.coco
import math
import os
import numpy as np
import traitlets

with open('preprocess/hand_pose.json', 'r') as f:
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

if not os.path.exists('model/hand_pose_resnet18_att_244_244_trt.pth'):
    MODEL_WEIGHTS = 'model/hand_pose_resnet18_att_244_244.pth'
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    import torch2trt
    print(model)
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

'''
OPTIMIZED_MODEL = 'model/hand_pose_resnet18_att_244_244_trt.pth'
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology,cmap_threshold=0.15, link_threshold=0.15)
draw_objects = DrawObjects(topology)



import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

'''
