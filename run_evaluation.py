"""
The evaluation entry point for WIDER Challenge 2019: Face Detection Accuracy+Runtime Track.

It will be the entrypoint for the evaluation docker once built.
Basically It downloads a list of images and run the face detector on each image.
Then the runtime and detection output will be reported to the evaluation system.

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
from eval_kit.client import upload_eval_output, get_image_iter, get_job_id


logging.basicConfig(level=logging.INFO)


########################################################################################################
# please change these lines to include your own face detector extending the eval_kit.detector.FaceDetector base class.
sys.path.append("mtcnn")
from sample_detector import SampleMTCNNFaceDetector as WIDERTestFaceDetectorClass
########################################################################################################


def evaluate_runtime(detector_class, image_iter, job_id):
    """
    Please DO NOT modify this part of code or the eval_kit
    Modification of the evaluation toolkit could result in cancellation of your award.

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

    logging.info("all image finished, uploading evaluation outputs for evaluation.")
    # send evaluation output to the server
    upload_eval_output(output_boxes, output_time, job_id)


if __name__ == '__main__':
    job_id = get_job_id()
    wider_test_image_iter = get_image_iter()
    evaluate_runtime(WIDERTestFaceDetectorClass, wider_test_image_iter, job_id)



