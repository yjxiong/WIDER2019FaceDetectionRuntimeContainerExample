# WIDER2019FaceDetectionRuntimeContainerExample
This repo provides an example docker container for runtime evaluation for the WIDER 2019 challenge track: face detection accuracy and runtime.

**Note**: We highly recommend the participants to test their docker images before submission. Please refer to [this section](#Test-the-docker-image-locally) for instructions.

# BEFORE YOU START: Request Resource Provision

First create an account on the [challenge website](https://competitions.codalab.org/competitions/22955) as well an [AWS account](https://aws.amazon.com/account/) (in any region except Beijing and Ningxia). Send your AWS account id (12 digits) and an email address to the organizing commitee's email address: `wider-challenge@ie.cuhk.edu.hk`. We will allocate evaluation resources for you.

# Obtain the the example

Run the following commands to build the example image

```bash

git clone https://github.com/yjxiong/WIDER2019FaceDetectionRuntimeContainerExample
cd WIDER2019FaceDetectionRuntimeContainerExample
docker build -t wider-challenge-<your_aws_id> .
```

# Include your own algorithm

- You algorithm should provide a Python class which implements the interfaces `FaceDetectorBase` in `eval_kit/detection.py`.
- Modify line 27 - 29 in `run_evalution.py` to import your own detector for evaluation.
- Build and submit the image to the ECR repo for your AWS account.

# Test the docker image locally

The submission process could take hours. It is slow to debug on the cloud. Here we provide some tools for the participants to locally test the correctness of the algorithms.

To verify the algorithm can run properly, run the following command
```bash
nvidia-docker run -it wider-challenge-<your_aws_id> python3 local_test.py
```
It will run the algorithms in the evaluation workflow on some sample images and print out the results.
You can compare your algorithm's output with the groundtruth for the sample images. 

The output will look like

```
    ================================================================================
    all image finished, showing verification info below:
    ================================================================================
    
INFO:root:Image ID: data/images/0--Parade/0_Parade_Parade_0_901.jpg, Runtime: 0.04310035705566406
INFO:root:       gt box: [7, 399, 201, 342]
INFO:root:       gt box: [283, 680, 164, 208]
INFO:root:       gt box: [712, 351, 292, 374]
INFO:root:       prediction box: [290.0172  685.2428  163.99304 210.28986   1.     ]
INFO:root:       prediction box: [671.7536     314.9785     340.5346     427.6707       0.99999464]
INFO:root:       prediction box: [ 16.449345  378.24097   279.26697   400.31653     0.9989442]
INFO:root: 
INFO:root:Image ID: data/images/7--Cheering/7_Cheering_Cheering_7_500.jpg, Runtime: 0.04457974433898926
INFO:root:       gt box: [442, 140, 312, 456]
INFO:root:       prediction box: [420.00894 144.01822 322.09244 405.97415   1.     ]
INFO:root: 
INFO:root:Image ID: data/images/0--Parade/0_Parade_Parade_0_829.jpg, Runtime: 0.11018848419189453
INFO:root:       gt box: [501, 160, 285, 443]
INFO:root:       prediction box: [496.86624 167.46695 334.38205 421.00534   1.     ]
INFO:root: 
INFO:root:Done. Average FPS: 15.162
```

# Submitting the docker image

Run the following commands (we assume you have set up the [Docker CLI authentication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/Registries.html#registry_auth))

Retrieve the login command to use to authenticate your Docker client to your registry.
Use the AWS CLI:

```bash
$(aws ecr get-login --no-include-email --region us-west-2 --registry-ids 624814826659)
```

Note: If you receive an "Unknown options: --no-include-email" error when using the AWS CLI, ensure that you have the latest version installed.

Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:

```bash
docker build -t wider-challenge-<your_aws_id> .
```

After the build completes, tag your image so you can push the image to the repository:

```bash
docker tag wider-challenge-<your_aws_id>:latest 624814826659.dkr.ecr.us-west-2.amazonaws.com/wider-challenge-<your_aws_id>:latest
```


Run the following command to push this image to your the AWS ECR repository:

```bash
docker push 624814826659.dkr.ecr.us-west-2.amazonaws.com/wider-challenge-<your_aws_id>:latest
```

After you pushed to the repo, the evaluation will automatically start. In **3 hours** you should receive a email with the evaluation results if the evaluation is successful.
