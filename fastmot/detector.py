from collections import defaultdict
from pathlib import Path
import configparser
import numpy as np
import numba as nb
import cv2

from . import models
from .utils import InferenceBackend
from .utils.rect import as_rect, to_tlbr, get_size, area
from .utils.rect import union, crop, multi_crop, iom, diou_nms
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import json
import torch
import trt_pose.coco
from pose.estimator import PoseEstimator
import math


DET_DTYPE = np.dtype(
    [('tlbr', float, 4),
     ('label', int),
     ('conf', float)],
    align=True
)


def pose_post(save_pose_img,  outputs):

    ori_w ,ori_h = 1920, 1080

    # 存放了各个人体的脖子,左右手
    results = list()

    # result = np.array(outputs).reshape((26, 24, 24))
    # result = np.transpose(result, (1, 2, 0))
    # output = result

    output =  np.array(outputs).reshape((24,24,26))
    humans = PoseEstimator.estimate(output[:, :, :10],output[:, :, 10:])

    fw = 24.0

    for human in humans:
        centerx, centery = 0, 0
        # pose 的左右手的位置,如果为0 ,则代表不存在．（被遮挡或者离开）中心点的位置
        ltx = 0
        lty = 0
        rbx = 0
        rby = 0

        # 脖子的位置 , 可能会丢，则采用两个肩膀的位置来取中间值判断
        if 0 in human.body_parts.keys():
            l2 = human.body_parts[0]
            x1, y1 = (l2.x + 0.5 / fw) * ori_w, (l2.y + 0.5 / fw) * ori_h
            centerx = int(x1)
            centery = int(y1)
            cv2.circle(save_pose_img, (centerx,centery), 8, (0,0,255), -1)

        # 通过两个肩膀去找脖子
        elif (1 in human.body_parts.keys()) and (0 not in human.body_parts.keys()) \
                and (5 in human.body_parts.keys()):
            l2_1 = human.body_parts[1]
            l2_2 = human.body_parts[5]
            x1_1, y1_1 = (l2_1.x + 0.5 / fw) * ori_w, (l2_1.y + 0.5 / fw) * ori_h
            x1_2, y1_2 = (l2_2.x + 0.5 / fw) * ori_w, (l2_2.y + 0.5 / fw) * ori_h
            centerx = (int(x1_1) + int(x1_2)) /2
            centery = (int(y1_1) + int(y1_2)) /2
            cv2.circle(save_pose_img, (centerx,centery), 20, (0, 255, 0), -1)

        if 1 in human.body_parts.keys() and 5 in human.body_parts.keys():
            l2_1 = human.body_parts[1]
            l2_2 = human.body_parts[5]
            x1_1, y1_1 = (l2_1.x + 0.5 / fw) * ori_w, (l2_1.y + 0.5 / fw) * ori_h
            x1_2, y1_2 = (l2_2.x + 0.5 / fw) * ori_w, (l2_2.y + 0.5 / fw) * ori_h
            cv2.line(save_pose_img,(int(x1_1),int(y1_1)),(centerx,centery),(0,0,255),3)#绿色，3个像素宽度
            cv2.line(save_pose_img,(int(x1_2),int(y1_2)),(centerx,centery),(0,0,255),3)#绿色，3个像素宽度  

        # 左手的位置
        if 4 in human.body_parts.keys():
            l2 = human.body_parts[4]
            ltx, lty = int((l2.x + 0.5 / fw) * ori_w), int((l2.y + 0.5 / fw) * ori_h)

        elif (3 in human.body_parts.keys()) and (4 not in human.body_parts.keys()):
            lhe = human.body_parts[2]
            lhw = human.body_parts[3]
            x1, y1 = (lhe.x + 0.5 / fw) * ori_w, (lhe.y + 0.5 / fw) * ori_h
            x2, y2 = (lhw.x + 0.5 / fw) * ori_w, (lhw.y + 0.5 / fw) * ori_h
            delx = x2 - x1
            dely = y2 - y1
            dist = math.sqrt(delx ** 2 + dely ** 2)
            norm_x, norm_y = delx * 1.0 / math.sqrt(delx ** 2 + dely ** 2), \
                                dely * 1.0 / math.sqrt(delx ** 2 + dely ** 2)
            new_x = x2 + 0.33 * dist * norm_x
            new_y = y2 + 0.33 * dist * norm_y
            circleX, circleY = (int(new_x), int(new_y))
            # circleX, circleY = self.border_check(circleX, circleY, ori_w, ori_h)
            ltx, lty = circleX, circleY

        # 右手的位置
        if 8 in human.body_parts.keys():
            r2 = human.body_parts[8]
            rbx, rby = (r2.x + 0.5 / fw) * ori_w, (r2.y + 0.5 / fw) * ori_h

        elif (7 in human.body_parts.keys()) and (8 not in human.body_parts.keys()):
            rhe = human.body_parts[6]
            rhw = human.body_parts[7]
            x1, y1 = (rhe.x + 0.5 / fw) * ori_w, (rhe.y + 0.5 / fw) * ori_h
            x2, y2 = (rhw.x + 0.5 / fw) * ori_w, (rhw.y + 0.5 / fw) * ori_h
            delx = x2 - x1
            dely = y2 - y1
            dist = math.sqrt(delx ** 2 + dely ** 2)
            norm_x, norm_y = delx * 1.0 / math.sqrt(delx ** 2 + dely ** 2), \
                                dely * 1.0 / math.sqrt(delx ** 2 + dely ** 2)
            new_x = x2 + 0.33 * dist * norm_x
            new_y = y2 + 0.33 * dist * norm_y
            circleX, circleY = (int(new_x), int(new_y))
            # circleX, circleY = self.border_check(circleX, circleY, ori_w, ori_h)

            rbx, rby = circleX, circleY

        if rbx > 0 and rby >0: 
            cv2.circle(save_pose_img, (int(rbx),int(rby)), 20, (0, 0, 255), -1)
        if ltx > 0 and lty >0: 
            cv2.circle(save_pose_img, (int(ltx),int(lty)), 20, (0, 0, 255), -1)

        # # if  centerx > 0 and centery > 0 and ( (rbx > 0 and rby > 0) or (ltx > 0 and lty > 0)):
        # #     results.append([(centerx, centery),(ltx,lty), (rbx, rby), idx])

        # print(rbx, rby, ltx, lty)
        # cv2.imwrite("out.jpg", save_pose_img)


class Detector:
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_id, frame, show_frame):
        self.detect_async(frame_id, frame)
        return self.postprocess(show_frame)

    def detect_async(self, frame_id, frame):
        """
        Asynchronous detection.
        """
        raise NotImplementedError

    def postprocess(self):
        """
        Synchronizes, applies postprocessing, and returns a record array
        of detections (DET_DTYPE).
        This function should be called after `detect_async`.
        """
        raise NotImplementedError


class YoloDetector(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.model = getattr(models, config['model'])
        self.class_ids = config['class_ids']
        self.conf_thresh = config['conf_thresh']
        self.max_area = config['max_area']
        self.nms_thresh = config['nms_thresh']

        self.backend = InferenceBackend(self.model, 1)
        self.input_handle, self.upscaled_sz, self.bbox_offset = self._create_letterbox()

    def detect_async(self, frame_id, frame):
        self._preprocess(frame)
        self.backend.infer_async()

    def postprocess(self, show_frame):
        det_out = self.backend.synchronize()
        det_out = np.concatenate(det_out).reshape(-1, 7)
        detections = self._filter_dets(det_out, self.upscaled_sz, self.class_ids, self.conf_thresh,
                                       self.nms_thresh, self.max_area, self.bbox_offset)
        detections = np.asarray(detections, dtype=DET_DTYPE).view(np.recarray)
        return detections

    def _preprocess(self, frame):
        frame = cv2.resize(frame, self.input_handle.shape[:0:-1])
        self._normalize(frame, self.input_handle)

    def _create_letterbox(self):
        src_size = np.asarray(self.size)
        #print('src_size: ', src_size.shape)
        #print('self.model: ', self.model.keys())
        #for i in self.modek.keys():
        #    print(self.model[i])
        dst_size = np.asarray(self.model.INPUT_SHAPE[:0:-1])

        if self.model.LETTERBOX:
            scale_factor = min(dst_size / src_size)
            scaled_size = np.rint(src_size * scale_factor).astype(int)
            img_offset = (dst_size - scaled_size) / 2
            insert_roi = to_tlbr(np.r_[img_offset, scaled_size])
            upscaled_sz = np.rint(dst_size / scale_factor).astype(int)
            bbox_offset = (upscaled_sz - src_size) / 2
            self.backend.input_handle = 0.5
        else:
            upscaled_sz = src_size
            insert_roi = to_tlbr(np.r_[0, 0, dst_size])
            bbox_offset = np.zeros(2)

        input_handle = self.backend.input_handle.reshape(self.model.INPUT_SHAPE)
        input_handle = crop(input_handle, insert_roi, chw=True)
        #print("input_handle size : ", input_handle.shape)
        #print("self.model.inputshape: ", self.model.INPUT_SHAPE)
        return input_handle, upscaled_sz, bbox_offset

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _normalize(frame, out):
        # BGR to RGB
        rgb = frame[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize to [0, 1] interval
        normalized = chw / 255.
        out[:] = normalized

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _filter_dets(det_out, size, class_ids, conf_thresh, nms_thresh, max_area, offset):
        """
        det_out: a list of 3 tensors, where each tensor
                 contains a multiple of 7 float32 numbers in
                 the order of [x, y, w, h, box_confidence, class_id, class_prob]
        """
        # drop detections with low score
        scores = det_out[:, 4] * det_out[:, 6]
        keep = np.where(scores >= conf_thresh)
        det_out = det_out[keep]

        # scale to pixel values
        det_out[:, :4] *= np.append(size, size)
        det_out[:, :2] -= offset

        keep = []
        for class_id in class_ids:
            class_idx = np.where(det_out[:, 5] == class_id)[0]
            class_dets = det_out[class_idx]
            class_keep = diou_nms(class_dets[:, :4], class_dets[:, 4], nms_thresh)
            keep.extend(class_idx[class_keep])
        keep = np.asarray(keep)
        nms_dets = det_out[keep]

        detections = []
        for i in range(len(nms_dets)):
            tlbr = to_tlbr(nms_dets[i, :4])
            # clip inside frame
            tlbr = np.maximum(tlbr, 0)
            tlbr = np.minimum(tlbr, np.append(size, size))
            label = int(nms_dets[i, 5])
            conf = nms_dets[i, 4] * nms_dets[i, 6]
            if 0 < area(tlbr) <= max_area:
                detections.append((tlbr, label, conf))
        return detections


# # train vgg pose
# class PoseNet(Detector):
#     def __init__(self, size, config):
#         super().__init__(size)
#         self.model = getattr(models, config['model'])
#         self.batch_size = 1

#         self.inp_stride = np.prod(self.model.INPUT_SHAPE)
#         self.backend = InferenceBackend(self.model, 1)
#         self.input_handle = self.backend.input_handle.reshape(self.model.INPUT_SHAPE)

#     def detect_async(self, frame_id, frame ):
        
#         frame = cv2.resize(frame, (192,192))
#         self._normalize(frame, self.input_handle)
#         self.backend.infer_async()

#     @staticmethod
#     @nb.njit(fastmath=True, cache=True)
#     def _normalize(frame, out):
#         rgb = frame[..., ::-1]
#         out[:] = rgb

#     def postprocess(self, show_frame):
#         det = self.backend.synchronize()
#         result = np.array(det[0]).reshape((24, 24, 26))

#         pose_post(show_frame, det[0])
#         return show_frame ,[]

# trt_pose model
class PoseNet(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.model = getattr(models, config['model'])
        self.batch_size = 1

        self.inp_stride = np.prod(self.model.INPUT_SHAPE)
        self.backend = InferenceBackend(self.model, 1)
        self.input_handle = self.backend.input_handle.reshape(self.model.INPUT_SHAPE)

        with open('fastmot/poselibs/human_pose.json', 'r') as f:
            human_pose = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)

    def detect_async(self, frame_id, frame ):
        
        frame = cv2.resize(frame, self.input_handle.shape[:0:-1])
        frame = frame.astype(np.float32) 
        frame = ((frame / 255.0 ) -0.45 )/ 0.225
        self._normalize(frame, self.input_handle)
        self.backend.infer_async()

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _normalize(frame, out):
        # BGR to RGB
        rgb = frame[..., ::-1]
        # HWC -> CHW
        normalized = rgb.transpose(2, 0, 1)
        out[:] = normalized

    def postprocess(self, show_frame):
        det_out = self.backend.synchronize()
        pcm = det_out[0].reshape((-1,18,56,56))
        paf = det_out[1].reshape((-1,42,56,56))
        counts, objects, peaks = self.parse_objects(torch.from_numpy(pcm), torch.from_numpy(paf))
        res = self.draw_objects(show_frame, counts, objects, peaks, "pose")
        return show_frame, res


class HandNet(Detector):
    def __init__(self, size, config):
        super().__init__(size)
        self.model = getattr(models, config['model'])
        self.batch_size = 1

        self.inp_stride = np.prod(self.model.INPUT_SHAPE)
        self.backend = InferenceBackend(self.model, 1)
        self.input_handle = self.backend.input_handle.reshape(self.model.INPUT_SHAPE)

        with open('fastmot/poselibs/hand_pose.json', 'r') as f:
            human_hand = json.load(f)

        topology = trt_pose.coco.coco_category_to_topology(human_hand)
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)

    def detect_async(self, frame_id, frame):
        
        frame = cv2.resize(frame, self.input_handle.shape[:0:-1])
        frame = frame.astype(np.float32) 
        frame = ((frame / 255.0 ) -0.45 )/ 0.225
        self._normalize(frame, self.input_handle)
        self.backend.infer_async()

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _normalize(frame, out):
        # BGR to RGB
        rgb = frame[..., ::-1]
        # HWC -> CHW
        normalized = rgb.transpose(2, 0, 1)
        out[:] = normalized

    def postprocess(self, show_frame):
        det_out = self.backend.synchronize()
        pcm = det_out[0].reshape((-1,21,56,56))
        paf = det_out[1].reshape((-1,40,56,56))
        counts, objects, peaks = self.parse_objects(torch.from_numpy(pcm), torch.from_numpy(paf))
        print(counts, objects, peaks)
        _ = self.draw_objects(show_frame, counts, objects, peaks, "hand")

        return show_frame