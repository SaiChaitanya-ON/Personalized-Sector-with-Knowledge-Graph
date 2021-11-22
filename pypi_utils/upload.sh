#!/bin/bash

set -e

USAGE="upload.sh <options> <package_name>
if packange_name is skipped, it will be extracted from the pyproject.toml file
options:
  -d/--dry-run                : don't do the actual upload, just check whether it would succeed
  -h/--help                   : print this
  --allow-unpublished-changes : allow unpublished changes
"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      printf "%s" "$USAGE"
      exit 0
      ;;
    -d|--dry-run)
      shift
      export IS_DRY_RUN=TRUE
      ;;
    --allow-unpublished-changes)
      shift
      export ALLOW_UNPUBLISHED=TRUE
      ;;
    *)
      break
      ;;
  esac
done

PACKAGE_NAME=$1
if [ -z $PACKAGE_NAME ]; then
    echo "package name not provided, taking it from pyproject.toml"
    PACKAGE_NAME=$(poetry version | cut -d' ' -f1)
fi

poetry config repositories.oaknorth https://oaknorth-264571470104.d.codeartifact.eu-west-1.amazonaws.com/pypi/python/
poetry config http-basic.oaknorth aws `aws codeartifact get-authorization-token --domain oaknorth --domain-owner 264571470104 --query authorizationToken --output text  --duration-seconds 3600 --region eu-west-1`

poetry build
# get the local hash of the artifacts we just build
LOCAL_VERSION=$(poetry version | cut -d' ' -f2-)
LOCAL_HASH=$(poetry run pip hash dist/*-py3-none-any.whl | grep "\-\-hash" | sed -e "s/^--hash=sha256://")

echo "local hash: $LOCAL_HASH"
echo "local version: $LOCAL_VERSION"

VERSION_RESULT=$(aws codeartifact list-package-versions --region eu-west-1 --domain oaknorth --domain-owner 264571470104 --repository python --format pypi --package $PACKAGE_NAME)
VERSIONS=$(echo $VERSION_RESULT | jq '.versions[].version' | sed -e 's/^"//' -e 's/"$//')
LATEST_PUBLISHED_VERSION=$(echo $VERSIONS | tr " " "\n" | sort -V | tail -n 1)
LOCAL_VERSION_EXISTS_REMOTELY=$(echo $VERSIONS | tr " " "\n" | grep -oE "^${LOCAL_VERSION}$" || echo "")

echo "published versions: $VERSIONS"
echo "latest published version: $LATEST_PUBLISHED_VERSION"

# read: LOCAL_VERSION > LATEST_PUBLISHED_VERSION
# see: https://unix.stackexchange.com/questions/285924/how-to-compare-a-programs-version-in-a-shell-script
if [ ! "$(printf '%s\n' "$LATEST_PUBLISHED_VERSION" "$LOCAL_VERSION" | sort -V | head -n1)" = "$LOCAL_VERSION" ]; then
    echo "local version is ahead of remote"
    if [ -z ${IS_DRY_RUN} ]; then # this is not a dry run
        poetry publish -r oaknorth
    else
        echo "dry run: publish would succeed"
    fi
    exit 0
elif [ ! -z "$LOCAL_VERSION_EXISTS_REMOTELY" ]; then
    echo "local version has already been published"
    # get the remote hash and version from codeartifact
    RESULT=$(aws codeartifact list-package-version-assets --region eu-west-1 --domain oaknorth --domain-owner 264571470104 --repository python --format pypi --package $PACKAGE_NAME --package-version $LOCAL_VERSION || {})
    REMOTE_HASH=$(echo $RESULT | jq '.assets[] | select(.name | endswith("-py3-none-any.whl")) | .hashes["SHA-256"]' | sed -e 's/^"//' -e 's/"$//')
    echo "remote hash: $REMOTE_HASH"

    if [ "$LOCAL_HASH" == "$REMOTE_HASH" ]; then
        echo "local and remote version is the same, and the code matches"
        exit 0
    fi
fi

if [ -z "$ALLOW_UNPUBLISHED" ]; then
  echo "There are unpublished code changes, but the local version is behind the published version. "
  echo "Please bump your version so that it is ahead of $LATEST_PUBLISHED_VERSION "
  echo "by running 'poetry version <major|minor|patch>'"
  exit 1
fi
