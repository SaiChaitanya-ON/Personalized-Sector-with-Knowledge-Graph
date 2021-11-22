#!/bin/bash

name=${1:-""}
CLUST_OR_INSTANCES=${2:-""}

if [ ! -z "$CLUST_OR_INSTANCES" ]; then
    if [[ ${CLUST_OR_INSTANCES} =~ j-[A-Z0-9]+ ]]; then
        CLUSTER_ID="$CLUST_OR_INSTANCES"
        MASTER="m4.2xlarge"
        CORE="m4.xlarge"
    else
        MASTER="$(cut -d, -f1 <<< $CLUST_OR_INSTANCES)"
        CORE="$(cut -d, -f2 <<< $CLUST_OR_INSTANCES)"
        NCORE="$(cut -d, -f3 <<< $CLUST_OR_INSTANCES)"
        if [ -z $"CORE" ]; then
            CORE="m4.xlarge"
        fi
        if [ -z $"MASTER" ]; then
            MASTER="m4.2xlarge"
        fi
        if [ -z $"NCORE" ]; then
            NCORE="2"
        fi
    fi
else
    MASTER="m4.2xlarge"
    CORE="m4.xlarge"
    NCORE="2"
fi


if [ -z "$CLUSTER_ID" ]; then
    cat emr/bootstrap_python.sh | sed -e "s/THIS_IS_A_LONG_MAKER_NAME/${name}/g" | aws s3 cp - s3://dev-space-eu-west-1/configs/${name}/bootstrap_python.sh
    aws s3 cp poetry.lock s3://dev-space-eu-west-1/configs/${name}/
    aws s3 cp pyproject.toml s3://dev-space-eu-west-1/configs/${name}/

    echo "Provisioning new cluster for ${name} with 1x $MASTER and ${NCORE}x $CORE"
    aws emr create-cluster \
    --applications Name=Spark Name=Hadoop Name=Zeppelin Name=JupyterHub Name=Livy \
    --bootstrap-actions "[{\"Path\":\"s3://dev-space-eu-west-1/configs/${name}/bootstrap_python.sh\",\"Name\":\"Prepare PySpark\"}]" \
    --ebs-root-volume-size 10 \
    --ec2-attributes '{"KeyName":"ec2eu1-ml-keypair","AdditionalSlaveSecurityGroups":["sg-05f91949264cdd8c8"],"InstanceProfile":"ml-emr-ec2-role","SubnetId":"subnet-0399c74b","EmrManagedSlaveSecurityGroup":"sg-05f91949264cdd8c8","EmrManagedMasterSecurityGroup":"sg-05f91949264cdd8c8","AdditionalMasterSecurityGroups":["sg-05f91949264cdd8c8"]}' \
    --service-role ml-emr-role \
    --release-label emr-5.24.1 --log-uri 's3n://aws-emr-1574078139-logs/' \
    --name "MLEng - $name" \
    --tags 'MLEng' \
    --instance-groups "[{\"InstanceCount\":1,\"EbsConfiguration\":{\"EbsBlockDeviceConfigs\":[{\"VolumeSpecification\":{\"SizeInGB\":64,\"VolumeType\":\"gp2\"},\"VolumesPerInstance\":1}]},\"InstanceGroupType\":\"MASTER\",\"InstanceType\":\"${MASTER}\",\"Name\":\"Master\"},{\"InstanceCount\":${NCORE},\"BidPrice\":\"OnDemandPrice\",\"EbsConfiguration\":{\"EbsBlockDeviceConfigs\":[{\"VolumeSpecification\":{\"SizeInGB\":64,\"VolumeType\":\"gp2\"},\"VolumesPerInstance\":1}]},\"InstanceGroupType\":\"CORE\",\"InstanceType\":\"${CORE}\",\"Name\":\"Core - 2\"}]" \
    --scale-down-behavior TERMINATE_AT_TASK_COMPLETION \
    --configurations file://emr/emr_config.json \
    --region eu-west-1 > emr/cluster_id.json
    python emr/populate_livy_conf.py
else
    echo "Using existing cluster $CLUSTER_ID"
    python emr/populate_livy_conf.py $CLUSTER_ID
fi
