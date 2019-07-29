# This is the sample Dockerfile for the evaluation.
# You can change it to include your own environment for submission.

FROM mxnet/python:1.4.1_gpu_cu90_py3

RUN pip3 install easydict boto3 opencv-python pillow

RUN apt update
RUN apt-get install -y -q libfontconfig1 libxrender1 libglib2.0-0 libsm6 libxext6 ucspi-tcp git

WORKDIR /app/face_det_eval
RUN git clone https://github.com/yjxiong/mtcnn

ADD . /app/face_det_eval

ENV MXNET_CUDNN_AUTOTUNE_DEFAULT=0

# This command runs the evaluation tool.
# DO NOT MODIFY THE LINE BELOW OTHERWISE THE EVALUATION WILL NOT RUN
CMD python3 run_evaluation.py