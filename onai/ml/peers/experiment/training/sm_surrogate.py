import argparse
import getpass
import logging
import os
from urllib.parse import urlparse

import boto3
import sagemaker
from sagemaker import s3_input
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.inputs import FileSystemInput

from onai.ml.tools.argparse import add_bool_argument
from onai.ml.tools.logging import setup_logger
from onai.ml.tools.sagemaker import get_private_subnets, get_sm_sg_id, get_sm_vpc_id
from onai.ml.tools.sagemaker.on_estimator import ONEstimator

logger = logging.getLogger(__name__)
setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--instance_type", default="ml.p3.2xlarge")
parser.add_argument("--current_user", default=getpass.getuser())
add_bool_argument(parser, "detach", default=True)
parser.add_argument(
    "-i", default="efs://fs-c18a2d0b/iat.chan@oaknorth.ai/surrogate_ds_full"
)
parser.add_argument("--name", default="surrogate-trainer")


args, remained_args = parser.parse_known_args()


bucket = "oaknorth-ml-dev-eu-west-1"
prefix = f"/{args.current_user}/sm"
output_dir = "s3://" + bucket + prefix

boto_session = boto3.Session()
sagemaker_client = boto_session.client("sagemaker")
sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_runtime_client=sagemaker_client,
    default_bucket=bucket,
)

subnets = [subnet.id for vpc in get_sm_vpc_id() for subnet in get_private_subnets(vpc)]

sg_ids = get_sm_sg_id()

logger.info("Subnets: %s. SGs: %s", subnets, sg_ids)


cmd_args = [
    "m.training_ds=/opt/ml/input/data/training/train",
    "m.val_ds=/opt/ml/input/data/training/val",
    "m.cuda=True",
] + remained_args


estimator = ONEstimator(
    entry_point="onai.ml.peers.experiment.training.surrogate",
    role="SagemakerService",
    train_instance_count=1,
    train_instance_type=args.instance_type,
    source_pkg="s3://oaknorth-ml-dev-eu-west-1/iat/code/onaiml.tar.gz",
    sagemaker_session=sagemaker_session,
    output_path=output_dir,
    subnets=subnets,
    security_group_ids=sg_ids,
    tags=[
        {"Key": "Creator", "Value": args.current_user},
        {"Key": "Project", "Value": "PEERS"},
    ],
    cmd_args=cmd_args,
    base_job_name=args.name,
    train_use_spot_instances=True,
    train_max_wait=3 * 24 * 60 * 60,
    train_max_run=3 * 24 * 60 * 60,
    train_volume_size=50,
    add_user_efs=True,
)


inputs = urlparse(args.i)

if inputs.scheme == "efs":
    inputs = {"training": FileSystemInput(inputs.netloc, "EFS", inputs.path, "ro")}
elif inputs.scheme == "s3":
    inputs = {"training": s3_input(s3_data=args.i, input_mode="File")}


estimator.fit(inputs, wait=not args.detach)
