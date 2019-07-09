# WIDER2019FaceDetectionRuntimeContainerExample
This repo provides an example docker container for runtime evaluation for the WIDER 2019 challenge track: face detection accuracy and runtime.

# BEFORE YOU START: Request Resource Provision

First create an account on the [challenge website](https://competitions.codalab.org/competitions/22955) as well an AWS account (in any region except Ningxia). Send your AWS account id (12 digits) and an address to the organizing commitee's email address: `wider-challenge@ie.cuhk.edu.hk`. We will allocate evaluation resources for you.

# Obtain the the example

Run the following commands to build the example image

```bash

git clone https://github.com/yjxiong/WIDER2019FaceDetectionRuntimeContainerExample
cd WIDER2019FaceDetectionRuntimeContainerExample
docker build -t wider-eval-example .
```

# Include your own algorithm

- You algorithm should provide a Python class which implements the interfaces `FaceDetectorBase` in `eval_kit/detection.py`.
- Modify line 27 - 29 in `run_evalution.py` to import your own detector for evaluation.
- Build and submit the image to the ECR repo for your AWS account.

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
