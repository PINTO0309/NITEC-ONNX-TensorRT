# NITEC-ONNX-TensorRT
ONNX implementation of "NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction" https://github.com/thohemp/nitec

## 1. Test

```
### The model is automatically downloaded on the first run.

python demo_nitec_onnx_tflite.py -v 0

python demo_nitec_onnx_tflite.py -v xxxx.mp4
```

- Base models

  https://github.com/PINTO0309/PINTO_model_zoo/tree/main/459_YOLOv9-Wholebody25

- TensorRT RTX3070 (YOLOv9 + NITEC)

  https://github.com/user-attachments/assets/09b646e5-def9-44a8-b1c2-2e6ea0a24558

- CPU Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz (YOLOv9 + NITEC)

  https://github.com/user-attachments/assets/e40a646a-711b-47d3-9e15-204803d33b47

```
usage: demo_nitec_onnx_tflite.py \
[-h]
[-odm {yolov9_n_wholebody25_post_0100_1x3x480x640.onnx}]
[-nim NITEC_MODEL]
[-v VIDEO]
[-ep {cpu,cuda,tensorrt}]
[-ost OBJECT_SOCRE_THRESHOLD]
[-ast ATTRIBUTE_SOCRE_THRESHOLD]
[-dvw]

options:
  -h, --help
    show this help message and exit
  -odm {yolov9_n_wholebody25_post_0100_1x3x480x640.onnx}, \
    --object_detection_model {yolov9_n_wholebody25_post_0100_1x3x480x640.onnx}
    ONNX/TFLite file path for YOLOX.
  -nim NITEC_MODEL, --nitec_model NITEC_MODEL
    ONNX/TFLite file path for NITEC.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -ost OBJECT_SOCRE_THRESHOLD, --object_socre_threshold OBJECT_SOCRE_THRESHOLD
    The detection score threshold for object detection. Default: 0.35
  -ast ATTRIBUTE_SOCRE_THRESHOLD, --attribute_socre_threshold ATTRIBUTE_SOCRE_THRESHOLD
    The attribute score threshold for object detection. Default: 0.70
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic recording to MP4.
    Devices that use a MicroSD card or similar for main storage can speed up overall processing.
```

## 2. Acknowledgements

1. https://github.com/thohemp/nitec

## 3. Convert to LiteRT (TFLite) / TensorFlow.js
```bash
onnx2tf \
-i nitec_rs18_e20_Nx3x224x224.onnx \
-cotof \
-coion

tensorflowjs_converter \
--input_format tf_saved_model \
--output_format tfjs_graph_model \
saved_model \
tfjs_model
```

## 4. Convert to OpenVINO IR
- Install OpenVINO

  https://www.isus.jp/wp-content/uploads/openvino/2024/docs/get-started/install-openvino.html?VERSION=v_2024_0_0&OP_SYSTEM=LINUX&DISTRIBUTION=APT

- Convert
  - https://www.isus.jp/wp-content/uploads/openvino/2024/docs/openvino-workflow/model-preparation.html
  - https://www.isus.jp/wp-content/uploads/openvino/2024/docs/openvino-workflow/model-preparation/convert-model-to-ir.html
    ```
    ovc nitec_rs18_e20_Nx3x224x224.onnx --output_model openvino/
    ovc yolov9_n_wholebody25_post_0100_1x3x480x640.onnx --output_model openvino/

    benchmark_app \
    -m openvino/nitec_rs18_e20_Nx3x224x224.xml \
    -b 1
    
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            16092 iterations
    [ INFO ] Duration:         60025.50 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        14.78 ms
    [ INFO ]    Average:       14.81 ms
    [ INFO ]    Min:           10.72 ms
    [ INFO ]    Max:           46.62 ms
    [ INFO ] Throughput:   268.09 FPS
    
    benchmark_app \
    -m openvino/yolov9_n_wholebody25_post_0100_1x3x480x640.xml \
    -shape "input_bgr[1,3,480,640]"
    
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            16816 iterations
    [ INFO ] Duration:         60014.17 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        14.09 ms
    [ INFO ]    Average:       14.18 ms
    [ INFO ]    Min:           9.72 ms
    [ INFO ]    Max:           31.78 ms
    [ INFO ] Throughput:   280.20 FPS
    ```
