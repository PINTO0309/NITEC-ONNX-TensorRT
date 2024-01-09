#!/bin/bash

# e.g. ./optimize_retinaface_tensorrt_engine.sh retinaface_resnet50_with_postprocess_Nx3x96x96_max001_th015.onnx

input_file="$1"

if [[ $input_file =~ Nx[0-9]+x([0-9]+)x([0-9]+) ]]; then
    H=${BASH_REMATCH[1]}
    W=${BASH_REMATCH[2]}
else
    echo "Error: Incorrect file name."
    exit 1
fi

start_time=$(date +%s)

for i in {1..20}; do
    sit4onnx -if "${input_file}" -b ${i}
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "elapsed_time: $elapsed sec"
