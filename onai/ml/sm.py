import json
import logging
import subprocess
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)


def get_private_subnets(vpc):
    for subnet in vpc.subnets.all():
        subnet_name = ([t["Value"] for t in subnet.tags if t["Key"] == "Name"] or [""])[
            0
        ]
        if "Private" in subnet_name:
            yield subnet


def get_sm_vpc_id():
    ec2 = boto3.resource("ec2", region_name="eu-west-1")
    return ec2.vpcs.filter(Filters=[{"Name": "tag:Name", "Values": ["berry-test"]}])


def get_sm_sg_id():
    client = boto3.client("ec2")
    resp = client.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": ["berry-test-sage-maker"]}]
    )
    return [r["GroupId"] for r in resp["SecurityGroups"]]


def get_smbase_image_id_from_garden():
    return json.loads(
        subprocess.check_output("garden get modules smbase-build -o json".split(" "))
    )["result"]["modules"]["smbase-build"]["outputs"]["deployment-image-id"]


def get_smbase_image_version_from_garden():
    return json.loads(
        subprocess.check_output("garden get modules smbase-build -o json".split(" "))
    )["result"]["modules"]["smbase-build"]["version"]["versionString"]


def get_sm_venv_path_from_garden():
    ver = json.loads(
        subprocess.check_output("garden get modules smbase-build -o json".split(" "))
    )["result"]["modules"]["smbase-build"]["version"]["versionString"]
    return urlparse(f"efs://fs-c18a2d0b/sm_poetry_cache/{ver}")
