#!/bin/bash
set -euo pipefail

if ! command -v aws &> /dev/null
then
    echo "aws cli is missing, please make sure it exists"
    exit 1
fi

if ! command -v poetry &> /dev/null
then
    echo "poetry is missing, please make sure it exists"
    exit 1
fi

export CODEARTIFACT_URL="oaknorth-264571470104.d.codeartifact.eu-west-1.amazonaws.com/pypi/python/simple"
export CODEARTIFACT_TOKEN=`aws codeartifact get-authorization-token --domain oaknorth --domain-owner 264571470104 --query authorizationToken --output text --region eu-west-1`
export CODEARTIFACT_USER=aws
poetry config repositories.oaknorth "https://$CODEARTIFACT_URL"
poetry config http-basic.oaknorth $CODEARTIFACT_USER $CODEARTIFACT_TOKEN
pip config set global.extra-index-url https://$CODEARTIFACT_USER:$CODEARTIFACT_TOKEN@$CODEARTIFACT_URL
