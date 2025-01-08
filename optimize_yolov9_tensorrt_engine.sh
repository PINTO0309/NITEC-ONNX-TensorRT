#!/bin/bash

# e.g. ./optimize_yolov9_tensorrt_engine.sh yolov9_n_wholebody25_post_0100_1x3x480x640.onnx

input_file="$1"

start_time=$(date +%s)

sit4onnx -if "${input_file}"  -fs 1 3 480 640

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "elapsed_time: $elapsed sec"
