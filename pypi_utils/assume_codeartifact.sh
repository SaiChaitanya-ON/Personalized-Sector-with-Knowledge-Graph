#!/usr/bin/env bash
CRED_FILE=$(mktemp)

aws sts assume-role --role-arn "arn:aws:iam::264571470104:role/garden-develop-codeartifact-user-access-role" --role-session-name CircleCI > $CRED_FILE

export AWS_SECRET_ACCESS_KEY=$(cat $CRED_FILE | jq --raw-output ".Credentials.SecretAccessKey")
export AWS_ACCESS_KEY_ID=$(cat $CRED_FILE | jq --raw-output ".Credentials.AccessKeyId")
export AWS_SESSION_TOKEN=$(cat $CRED_FILE | jq --raw-output ".Credentials.SessionToken")
