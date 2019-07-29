"""
This is an example of the face detector class which wil be called by the evaluation toolkit

The implementation is from
https://github.com/Seanlinx/mtcnn/blob/master/demo.py

"""


import numpy as np
import mxnet as mx
import argparse
import cv2
import time
from core.symbol import P_Net, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector import MtcnnDetector

from eval_kit.detector import FaceDetectorBase

class SampleMTCNNFaceDetector(FaceDetectorBase):

    def __init__(self):
        """
        This is an example face detector used for runtime evaluation in WIDER Challenge 2019.
        It takes one BGR image and output a numpy array of face boudning boxes in the format of
        [left, top, width, height, confidence]

        Your own face detector should also follow this interface and implement the FaceDetectorBase interface
        Please find the detailed requirement of the face detector in FaceDetectorBase
        """
        super(SampleMTCNNFaceDetector, self).__init__()
        self.detectors = [None, None, None]

        prefix = ['mtcnn/model/pnet', 'mtcnn/model/rnet', 'mtcnn/model/onet']
        epoch = [16, 16, 16]
        batch_size = [2048, 256, 16]
        slide_window = False
        min_face_size = 40
        stride = 2
        thresh = [0.5, 0.5, 0.7]

        # the evaluation environment provides one NVIDIA P100 GPU
        ctx = mx.gpu(0)

        # load pnet model
        args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
        if slide_window:
            PNet = Detector(P_Net("test"), 12, batch_size[0], ctx, args, auxs)
        else:
            PNet = FcnDetector(P_Net("test"), ctx, args, auxs)
        self.detectors[0] = PNet

        # load rnet model
        args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        self.detectors[1] = RNet

        # load onet model
        args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test"), 48, batch_size[2], ctx, args, auxs)
        self.detectors[2] = ONet

        self.mtcnn_detector = MtcnnDetector(detectors=self.detectors, ctx=ctx, min_face_size=min_face_size,
                                       stride=stride, threshold=thresh, slide_window=slide_window)

    def process_image(self, image):
        """
        :param image: a numpy.array representing one image with BGR colors
        :return: numpy.array of detection results in the format of
            [
                [left, top, width, height, confidence],
                ...
            ], dtype=np.float32
        """

        boxes, boxes_c = self.mtcnn_detector.detect_pnet(image)
        if boxes_c is None:
            return np.array([], dtype=np.float32)
        boxes, boxes_c = self.mtcnn_detector.detect_rnet(image, boxes_c)
        if boxes_c is None:
            return np.array([], dtype=np.float32)
        boxes, boxes_c = self.mtcnn_detector.detect_onet(image, boxes_c)
        if boxes_c is None:
            return np.array([], dtype=np.float32)
        raw_boxes = np.array(boxes_c, dtype=np.float32)
        raw_boxes[:, 2] = raw_boxes[:, 2] - raw_boxes[:, 0]
        raw_boxes[:, 3] = raw_boxes[:, 3] - raw_boxes[:, 1]
        return raw_boxes

