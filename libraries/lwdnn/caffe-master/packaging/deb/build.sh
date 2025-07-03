#!/bin/bash
set -e

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
SRC_DIR=$(dirname "$(dirname "$LOCAL_DIR")")
cd $SRC_DIR

################################################################################
### Check for basic requirements
################################################################################

if ! which git >/dev/null 2>&1; then
    echo "git not installed"
    exit 1
fi
if ! git rev-parse >/dev/null 2>&1; then
    echo "not a git repository"
    exit 1
fi
if [ "$(git rev-parse --show-toplevel)" != "$SRC_DIR" ]; then
    echo "$SRC_DIR is not a git repository"
    exit 1
fi
if ! which python >/dev/null 2>&1; then
    echo "python not installed"
    exit 1
fi
if ! git diff-index --quiet HEAD >/dev/null 2>&1; then
    echo "git index is dirty - either stash or commit your changes"
    exit 1
fi
if ! which docker >/dev/null 2>&1; then
    echo "docker not installed"
    exit 1
fi

################################################################################
# Read elwvars
################################################################################

if [ -z "$DOCKER_BASE" ]; then
    echo "DOCKER_BASE is a required environment variable"
    exit 1
fi
echo "DOCKER_BASE: $DOCKER_BASE"
DOCKER_BUILD_ID="caffe-lw-debuild"
DOCKER_BASE_TAG="caffe-lw-debuild-base"
docker tag ${DOCKER_BASE} ${DOCKER_BASE_TAG}
if [[ "$(docker run --rm ${DOCKER_BASE_TAG} bash -c "ldconfig -p | grep 'liblwdart.so ' | wc -l")" -ne 1 ]]; then
    echo "LWCA is not installed on this image"
    exit 1
fi
if [[ "$(docker run --rm ${DOCKER_BASE_TAG} bash -c "ldconfig -p | grep 'liblwdnn.so ' | wc -l")" -ne 1 ]]; then
    echo "lwDNN is not installed on this image"
    exit 1
fi
if [[ "$(docker run --rm ${DOCKER_BASE_TAG} bash -c "ldconfig -p | grep 'libnccl.so ' | wc -l")" -ne 1 ]]; then
    echo "LWCL is not installed on this image"
    exit 1
fi

if [ -z "$DEBIAN_REVISION" ]; then
    echo ">>> Using default DEBIAN_REVISION (set the elwvar to override)"
    OS_NAME=$(docker run --rm ${DOCKER_BASE_TAG} sh -c ". /etc/os-release && echo \$ID\$VERSION_ID")
    LWDA_VERSION=$(docker inspect -f '{{index .Config.Labels "com.lwpu.lwca.version"}}' ${DOCKER_BASE_TAG})
    if [[ -z "$LWDA_VERSION" ]]; then
        LWDA_VERSION=$(docker run --rm ${DOCKER_BASE_TAG} bash -c "readlink /usr/local/lwca-*/lib*/liblwdart.so | sort -u | cut -c14-")
    fi
    DEBIAN_REVISION=1${OS_NAME}+lwca${LWDA_VERSION}
fi
echo "DEBIAN_REVISION: $DEBIAN_REVISION"

################################################################################
# Callwlate versions
################################################################################

DESCRIBE_VERSION=$(git describe)
GIT_TAG=$(git describe --abbrev=0)
UPSTREAM_VERSION=${DESCRIBE_VERSION:1}
if [[ "$GIT_TAG" == *"-"* ]]; then
    # Replace the first dash with a tilde
    UPSTREAM_VERSION=${UPSTREAM_VERSION/-/\~}
fi
# Replace the first dash with a plus
UPSTREAM_VERSION=${UPSTREAM_VERSION/-/+}
# Replace all dashes with dots
UPSTREAM_VERSION=${UPSTREAM_VERSION//-/.}
echo UPSTREAM_VERSION: $UPSTREAM_VERSION
DEBIAN_VERSION=${UPSTREAM_VERSION}-${DEBIAN_REVISION}
echo DEBIAN_VERSION: $DEBIAN_VERSION

################################################################################
# Create source tarball
################################################################################

TARBALL_DIR="${LOCAL_DIR}/tarball/"
rm -rf $TARBALL_DIR
mkdir -p $TARBALL_DIR
git archive --prefix "caffe-lw/" -o $TARBALL_DIR/caffe-lw.orig.tar.gz HEAD

################################################################################
# Build
################################################################################

cd $LOCAL_DIR
docker build -t $DOCKER_BUILD_ID \
    --build-arg UPSTREAM_VERSION=$UPSTREAM_VERSION \
    --build-arg DEBIAN_VERSION=$DEBIAN_VERSION \
    --build-arg LWDA_VERSION=$LWDA_VERSION \
    .
docker ps -a -f "name=${DOCKER_BUILD_ID}" -q | xargs -r docker rm
docker create --name=$DOCKER_BUILD_ID $DOCKER_BUILD_ID
DIST_ROOT=$LOCAL_DIR/dist
DIST_DIR=$DIST_ROOT/$DEBIAN_VERSION
rm -rf $DIST_DIR
mkdir -p $DIST_ROOT
docker cp $DOCKER_BUILD_ID:/dist $DIST_DIR
docker rm $DOCKER_BUILD_ID
find $DIST_DIR -type f | sort
