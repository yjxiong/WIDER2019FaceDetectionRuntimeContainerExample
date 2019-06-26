import os
import time
import sys
import logging

import numpy as np
from eval_kit.client import upload_eval_output, get_image_iter


logging.basicConfig(level=logging.DEBUG)


########################################################################################################
# please change this your own face detector which extends the eval_kit.detector.FaceDetector base class.
sys.path.append("mtcnn")
from sample_detector import SampleMTCNNFaceDetector
########################################################################################################


def evaluate_runtime(detector_class, image_iter, job_id):
    """
    Please DO NOT modify this part of code or the eval_kit
    Modification of the evaluation toolkit could lead cancellation of your award.

    In this function we create the detector instance. And evaluate the wall time for running the WIDERFace test set.
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

    logging.info("all image finished, uploading evaluation outputs for evaluation.")
    # send evaluation output to the server
    upload_eval_output(output_boxes, output_time, job_id)


if __name__ == '__main__':
    job_id = "test"
    wider_test_image_iter = get_image_iter(max_number=10)
    evaluate_runtime(SampleMTCNNFaceDetector, wider_test_image_iter, "test")


