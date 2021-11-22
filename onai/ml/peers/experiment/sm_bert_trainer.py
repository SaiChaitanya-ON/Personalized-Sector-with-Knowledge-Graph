import argparse
import getpass
import logging

import boto3
import sagemaker

from onai.ml.tools.argparse import add_bool_argument
from onai.ml.tools.logging import setup_logger
from onai.ml.tools.sagemaker import (
    get_private_subnets,
    get_sm_sg_id,
    get_sm_vpc_id,
    get_smbase_image_id_from_garden,
    get_smbase_image_version_from_garden,
)
from onai.ml.tools.sagemaker.on_estimator import ONEstimator

logger = logging.getLogger(__name__)
setup_logger()

parser = argparse.ArgumentParser()

parser.add_argument("--instance_type", default="ml.g4dn.xlarge")
parser.add_argument("--current_user", default=getpass.getuser())
add_bool_argument(parser, "detach", default=True)
parser.add_argument("--name", default="peer-trainer")
args, remained_args = parser.parse_known_args()

bucket = "oaknorth-ml-dev-eu-west-1"
prefix = f"/{args.current_user}/sm"
output_dir = "s3://" + bucket + prefix
sm_local_output = "/opt/ml/checkpoints/"
checkpoint_s3_uri = f"{output_dir}/{sagemaker.utils.name_from_base('checkpoint')}"

boto_session = boto3.Session()
sagemaker_client = boto_session.client("sagemaker")
sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_runtime_client=sagemaker_client,
    default_bucket=bucket,
)

cmd_args = ["--local_ckpt", sm_local_output] + remained_args
subnets = [subnet.id for vpc in get_sm_vpc_id() for subnet in get_private_subnets(vpc)]

sg_ids = get_sm_sg_id()

logger.info("Subnets: %s. SGs: %s", subnets, sg_ids)
logger.info("checkpoints saved at : %s", checkpoint_s3_uri)

estimator = ONEstimator(
    entry_point="onai.ml.peers.experiment.training.bert_trainer",
    role="SagemakerService",
    train_instance_count=1,
    train_instance_type=args.instance_type,
    source_pkg="s3://oaknorth-ml-dev-eu-west-1/delan/code/onaiml.tar.gz",
    sagemaker_session=sagemaker_session,
    output_path=output_dir,
    subnets=subnets,
    security_group_ids=sg_ids,
    tags=[
        {"Key": "Creator", "Value": args.current_user},
        {"Key": "Project", "Value": "PEERS"},
    ],
    cmd_args=cmd_args,
    base_job_name="peer-trainer",
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path=sm_local_output,
    train_use_spot_instances=True,
    train_max_run=24 * 60 * 60,
    train_max_wait=48 * 60 * 60,
    add_hydra_run_dir=False,
    train_volume_size=50,
    add_user_efs=True,
    image_name=get_smbase_image_id_from_garden(),
    extra_python_path=f"/mnt/efs_root/sm_poetry_cache/"
    f"{get_smbase_image_version_from_garden()}/lib/python3.7/site-packages",
)

estimator.fit(wait=not args.detach)
