#!/bin/bash

# e.g. ./optimize_yolox_tensorrt_engine.sh yolox_x_body_head_hand_0102_0.5533_post_1x3x480x640.onnx

input_file="$1"

if [[ $input_file =~ Nx[0-9]+x([0-9]+)x([0-9]+) ]]; then
    H=${BASH_REMATCH[1]}
    W=${BASH_REMATCH[2]}
else
    echo "Error: Incorrect file name."
    exit 1
fi

start_time=$(date +%s)

sit4onnx -if "${input_file}"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "elapsed_time: $elapsed sec"
