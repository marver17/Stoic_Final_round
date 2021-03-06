#!/usr/bin/env bash

./build.sh
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# make the artifact folder writable

chmod 777 $SCRIPTPATH/../inference/artifact/

# Clear the artifact folder
rm -r $SCRIPTPATH/../inference/artifact/*

# Run the algorithm
MEMORY="32g"

docker run --rm --gpus all \
        --memory=$MEMORY --memory-swap=$MEMORY \
        --cap-drop=ALL --security-opt="no-new-privileges" \
        --network none --shm-size=1G --pids-limit 256 \
        -v /mnt/NAS_25/RAW/STOIC2021/:/input/ \
        -v $SCRIPTPATH/../inference/artifact/:/output/ \
        stoictrain
