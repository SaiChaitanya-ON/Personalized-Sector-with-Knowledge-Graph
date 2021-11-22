import argparse
import getpass
import logging

import boto3
import sagemaker

from onai.ml.tools.logging import setup_logger
from onai.ml.tools.sagemaker import get_private_subnets, get_sm_sg_id, get_sm_vpc_id
from onai.ml.tools.sagemaker.on_estimator import ONEstimator

logger = logging.getLogger(__name__)
setup_logger()

parser = argparse.ArgumentParser()

parser.add_argument("--instance_type", default="ml.g4dn.xlarge")
parser.add_argument("--current_user", default=getpass.getuser())
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


estimator = ONEstimator(
    entry_point="onai.ml.peers.experiment.surrogate_data_scorer",
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
    base_job_name="surrogate-data-scorer",
    cmd_args=remained_args,
    train_max_run=3 * 24 * 60 * 60,
    add_hydra_run_dir=False,
)

estimator.fit()
