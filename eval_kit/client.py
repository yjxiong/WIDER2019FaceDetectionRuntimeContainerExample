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

IMAGE_LIST_LAMBDA = 'WIDERImageList'
EVAL_LAMBDA = 'WIDEREval'
UPLOAD_BUCKET = 'wider-runtime-eval-dev'
UPLOAD_PREFIX = 'outputs'
UPLOAD_JSON_NAME = 'wider_output.json'

def _get_lambda_client():
    lambda_client = boto3.client('lambda',
                                 region_name='us-west-2',
                                 endpoint_url='https://lambda.us-west-2.amazonaws.com')
    return lambda_client


def _download_s3_image(s3_bucket, s3_path):
    s3_client = boto3.client('s3',
                             region_name='us-west-2',
                             endpoint_url='https://s3.us-west-2.amazonaws.com'
                             )

    f = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    return  cv2.imdecode(np.frombuffer(f.getvalue(), dtype=np.uint8), 1)

def _upload_output_to_s3(data, filename, s3_bucket, s3_path):
    s3_client = boto3.client('s3',
                             region_name='us-west-2',
                             endpoint_url='https://s3.us-west-2.amazonaws.com'
                             )
    json.dump(data, open(UPLOAD_JSON_NAME, 'w'))
    with zipfile.ZipFile(filename, 'w') as zf:
        zf.write(UPLOAD_JSON_NAME, compress_type=compression)

    s3_client.upload_file(filename, s3_bucket, s3_path)


def upload_eval_output(output_boxes, output_time, job_id):
    """
    This function uploads the testing output to a Lambda function to evaluate face detection AP.
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

    filename = '{}.zip'.format(job_id)
    s3_path = os.path.join(UPLOAD_PREFIX, filename)

    _upload_output_to_s3(upload_data, filename, UPLOAD_BUCKET, s3_path)

    response = _get_lambda_client().invoke(
        FunctionName=EVAL_LAMBDA,
        InvocationType='RequestResponse',
        Payload=json.dumps({
            's3_bucket':UPLOAD_BUCKET,
            's3_path': s3_path
        }),
    )

    logging.info(response['Payload'].read().decode('utf-8'))

def get_image_iter():
    """
    This function returns a iterator of input images for the detector
    Each iteration provides a tuple of
    (image_id: str, image: numpy.ndarray)
    the image will be in BGR color format with an array shape of (height, width, 3)
    :return: tuple(image_id, image)
    """
    request_payload = {
        "user_id": '123'
    }

    response_obj = _get_lambda_client().invoke(
        FunctionName=IMAGE_LIST_LAMBDA,
        InvocationType='RequestResponse',
        Payload=json.dumps(request_payload),
    )

    response = json.loads(response_obj['Payload'].read().decode('utf-8'))
    image_list = response['body']['s3_ids']
    s3_bucket = response['body']['s3_bucket']
    s3_prefix = response['body']['s3_prefix']

    logging.info("got image list, {} images".format(len(image_list)))

    for image_id in image_list:
        # get image from s3
        # decode image and convert to numpy

        st = time.time()
        image = _download_s3_image(s3_bucket, os.path.join(s3_prefix, image_id))
        elapsed = time.time() - st
        logging.info("image down time: {}".format(elapsed))
        yield image_id, image

