"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your face detection algorithm on several local images and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format (we expect (left, top, width, height) in pixels)

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong
Contact: bitxiong@gmail.com

WIDER Challenge 2019
"""

import time
import sys
import logging

import numpy as np
from eval_kit.client import get_local_image_iter, verify_local_output

logging.basicConfig(level=logging.INFO)

########################################################################################################
# please change these lines to include your own face detector extending the eval_kit.detector.FaceDetector base class.
sys.path.append("mtcnn")
from sample_detector import SampleMTCNNFaceDetector as WIDERTestFaceDetectorClass
########################################################################################################


def run_local_test(detector_class, image_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing face detection.
    """

    # initialize the detector
    logging.info("Initializing face detector.")
    try:
        detector = detector_class()
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")


    # run the images one-by-one and get runtime
    overall_time = 0
    output_boxes = {}
    output_time = {}
    eval_cnt = 0

    logging.info("Starting runtime evaluation")
    for image_id, image in image_iter:
        time_before = time.time()
        try:
            boxes = detector.process_image(image)
            assert isinstance(boxes, np.ndarray)
            output_boxes[image_id] = boxes
        except:
            # send errors to the eval frontend
            logging.error("Image id failed: {}".format(image_id))
            raise
        elapsed = time.time() - time_before
        output_time[image_id] = elapsed
        logging.info("image {} run time: {}".format(image_id, elapsed))

        overall_time += elapsed
        eval_cnt += 1

        if eval_cnt % 100 == 0:
            logging.info("Finished {} images".format(eval_cnt))

    logging.info("""
    ================================================================================
    all image finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    verify_local_output(output_boxes, output_time)


if __name__ == '__main__':
    local_test_image_iter = get_local_image_iter()
    run_local_test(WIDERTestFaceDetectorClass, local_test_image_iter)