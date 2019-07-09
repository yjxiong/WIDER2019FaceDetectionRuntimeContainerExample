import boto3
import json
import cv2
import os
import time

import logging
import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

import numpy as np

from io import BytesIO


# EVALUATION SYSTEM SETTINGS
# Do not modify these file otherwise your evaluation will fail.

WORKSPACE_BUCKET = 'wider2019-eval-workspace'
IMAGE_LIST_PATH = 'test-data/wider2019_face_detection_runtime_eval_image_list.txt'
IMAGE_PREFIX = 'test-images/'
UPLOAD_PREFIX = 'test-upload/'


def _get_s3_image_list(s3_bucket, s3_path):
    s3_client = boto3.client('s3',
                             region_name='us-west-2',
                             endpoint_url='https://s3.us-west-2.amazonaws.com'
                             )

    f = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    lines = f.getvalue().decode('utf-8').split('\n')
    return  [x.strip() for x in lines if x != '']


def _download_s3_image(s3_bucket, s3_path):
    s3_client = boto3.client('s3',
                             region_name='us-west-2',
                             endpoint_url='https://s3.us-west-2.amazonaws.com'
                             )

    f = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    return  cv2.imdecode(np.frombuffer(f.getvalue(), dtype=np.uint8),
                         cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)

def _upload_output_to_s3(data, filename, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3',
                             region_name='us-west-2',
                             endpoint_url='https://s3.us-west-2.amazonaws.com'
                             )
    local_path = '/tmp/{}'.format(filename)
    s3_path = os.path.join(s3_prefix, filename)
    json.dump(data, open(local_path, 'w'))
    s3_client.upload_file(local_path, s3_bucket, s3_path)


def get_job_id():
    return os.environ['WIDER_EVAL_JOB_ID']


def upload_eval_output(output_boxes, output_time, job_id):
    """
    This function uploads the testing output to S3 to trigger evaluation.
    :param output_boxes:
    :param output_time:
    :return:
    """
    upload_data = {
        k: {
            "boxes": output_boxes[k].tolist(),
            "runtime": output_time[k]
        } for k in output_boxes
    }

    filename = '{}.json'.format(job_id)

    _upload_output_to_s3(upload_data, filename, WORKSPACE_BUCKET, UPLOAD_PREFIX)

    logging.info("output uploaded to {}".format(filename))

def get_image_iter(max_number=None):
    """
    This function returns a iterator of input images for the detector
    Each iteration provides a tuple of
    (image_id: str, image: numpy.ndarray)
    the image will be in BGR color format with an array shape of (height, width, 3)
    :return: tuple(image_id, image)
    """

    image_list = _get_s3_image_list(WORKSPACE_BUCKET, IMAGE_LIST_PATH)

    if max_number is not None:
        image_list = image_list[:max_number]

    logging.info("got image list, {} images".format(len(image_list)))

    for image_id in image_list:
        # get image from s3
        # decode image and convert to numpy

        st = time.time()
        try:
            image = _download_s3_image(WORKSPACE_BUCKET, os.path.join(IMAGE_PREFIX, image_id))
        except:
            logging.info("Failed to download image: {}".format(os.path.join(IMAGE_PREFIX, image_id)))
            raise
        elapsed = time.time() - st
        logging.info("image down time: {}".format(elapsed))
        yield image_id, image

