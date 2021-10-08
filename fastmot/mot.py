from enum import Enum
import logging
import time
import cv2

from .detector import  YoloDetector, PoseNet, HandNet
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils.visualization import draw_tracks, draw_detections
from .utils.visualization import draw_flow_bboxes, draw_background_flow
import os
import random


LOGGER = logging.getLogger(__name__)


class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2


class MOT:
    """
    This is the top level module that integrates detection, feature extraction,
    and tracking together.
    Parameters
    ----------
    size : (int, int)
        Width and height of each frame.
    capture_dt : float
        Time interval in seconds between each captured frame.
    config : Dict
        Tracker configuration.
    draw : bool
        Flag to toggle visualization drawing.
    verbose : bool
        Flag to toggle output verbosity.
    """

    def __init__(self, size, capture_dt, config, draw=False, verbose=False):
        self.size = size
        self.draw = draw
        self.verbose = verbose
        self.detector_type = DetectorType[config['detector_type']]
        self.detector_frame_skip = config['detector_frame_skip']
        self.face_frame_skip = config['face_frame_skip']


        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, config['ssd_detector'])
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YoloDetector(self.size, config['yolo_detector'])
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, config['public_detector'])

        self.pose_model = PoseNet(self.size, config['pose_net'])
        self.hand_model = HandNet(self.size, config['hand_net'])

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(config['feature_extractor'])
        self.tracker = MultiTracker(self.size, capture_dt, self.extractor.metric,
                                    config['multi_tracker'])

        # reset counters
        self.frame_count = 0
        self.detector_frame_count = 0
        self.preproc_time = 0
        self.detector_time = 0
        self.extractor_time = 0
        self.association_time = 0
        self.tracker_time = 0

        if not os.path.exists("imgs"):
            os.mkdir("imgs")
            os.mkdir("imgs/hand_img")
            os.mkdir("imgs/detect_img")
            os.mkdir("imgs/pose_img")

    @property
    def visible_tracks(self):
        # retrieve confirmed and active tracks from the tracker
        return [track for track in self.tracker.tracks.values()
                if track.confirmed and track.active]

    def initiate(self):
        """
        Resets multiple object tracker.
        """
        self.frame_count = 0

    def step(self, frame):
        """
        Runs multiple object tracker on the next frame.
        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        show_frame = frame.copy()
        detect_frame = frame.copy()
        self.frame_count += 1

        # if self.frame_count < 3000 or self.frame_count > 5000:
        if  self.frame_count < 7000 or self.frame_count > 10000:
            return None

        # print("--------------", self.frame_count)
        detections = []
        if self.frame_count == 0:
            detections = self.detector(self.frame_count, frame, show_frame)
            self.tracker.initiate(frame, detections)
        else:

            if self.frame_count % self.face_frame_skip == 0:
                # pose_detect
                cv2.imwrite("imgs/pose_img/" + str(self.frame_count) + ".jpg", frame)

                self.pose_model.detect_async(self.frame_count, frame)
                show_frame , hand_imgs = self.pose_model.postprocess(frame)

                cv2.imwrite("imgs/pose_img/result_" + str(self.frame_count) + ".jpg", show_frame)

                if len(hand_imgs):
                    for img , x, y, roi_dis in hand_imgs:
                        # self.hand_model.detect_async(self.frame_count, img)

                        cv2.imwrite("imgs/hand_img/" + str(self.frame_count) +"_" +  str(random.random())+ ".jpg", img)

                        # img = cv2.imread('assets/test.jpg')
                        # for i in os.listdir("../assets/hand_pose/"):
                        #     img_name = os.path.join("../assets/hand_pose/", i)
                        #     img = cv2.imread(img_name)

                        # img = cv2.imread("../assets/2021-02-01_15-36-25_16829.jpg")
                        # self.hand_model.detect_async(self.frame_count, img)
                        # hand_frame = self.hand_model.postprocess(img)

                        # cv2.imwrite("imgs/hand_img/result_" + str(self.frame_count) + ".jpg", hand_frame)

                        # show_frame[y - roi_dis:y + roi_dis, x -roi_dis:x + roi_dis] = hand_frame

            if self.frame_count % self.detector_frame_skip == 0:
                tic = time.perf_counter()
                cv2.imwrite("imgs/detect_img/" + str(self.frame_count) + ".jpg", frame)

                self.detector.detect_async(self.frame_count, detect_frame)
                # print("detect : ", time.perf_counter() - tic)
                self.preproc_time += time.perf_counter() - tic
                #tic = time.perf_counter()
                #self.tracker.compute_flow(frame)
                #print("compute flow : ", time.perf_counter() - tic)
                detections = self.detector.postprocess(detect_frame)
                self.detector_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.extractor.extract_async(detect_frame, detections)
                self.tracker.apply_kalman()
                embeddings = self.extractor.postprocess()
                self.extractor_time += time.perf_counter() - tic
                tic = time.perf_counter()
                self.tracker.update(self.frame_count, detections, embeddings)
                self.association_time += time.perf_counter() - tic
                self.detector_frame_count += 1
            else:
                tic = time.perf_counter()
                self.tracker.track(detect_frame)
                self.tracker_time += time.perf_counter() - tic
                # print("only tracker : ", time.perf_counter() - tic)                

        if self.draw:
            # print("show", len(detections))
            self._draw(show_frame, detections)

        return show_frame

    def _draw(self, frame, detections):
        draw_tracks(frame, self.visible_tracks, show_flow=self.verbose)
        if self.verbose:
            draw_detections(frame, detections)
            draw_flow_bboxes(frame, self.tracker)
            draw_background_flow(frame, self.tracker)
        cv2.putText(frame, f'visible: {len(self.visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)