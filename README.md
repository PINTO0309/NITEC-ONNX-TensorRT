# NITEC-ONNX-TensorRT
ONNX implementation of "NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction" https://github.com/thohemp/nitec

## 1. Test

```
### The model is automatically downloaded on the first run.

docker pull pinto0309/botsort_onnx_tensorrt:latest

# With USBCam
xhost +local: && \
docker run --rm -it --gpus all \
-v `pwd`:/workdir \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
pinto0309/botsort_onnx_tensorrt:latest

# Without USBCam
xhost +local: && \
docker run --rm -it --gpus all \
-v `pwd`:/workdir \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
pinto0309/botsort_onnx_tensorrt:latest

python demo_nitec_onnx_tflite.py -v 0

python demo_nitec_onnx_tflite.py -v xxxx.mp4
```

https://github.com/PINTO0309/NITEC-ONNX-TensorRT/assets/33194443/3a8f6a9b-c7b1-4ffc-9d4c-d5f57734554e

```
usage: demo_nitec_onnx_tflite.py \
[-h]
[-odm {
    yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640.onnx,
    yolox_t_body_head_hand_post_0299_0.4522_1x3x480x640.onnx,
    yolox_s_body_head_hand_post_0299_0.4983_1x3x480x640.onnx,
    yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx,
    yolox_l_body_head_hand_0299_0.5420_post_1x3x480x640.onnx,
    yolox_x_body_head_hand_0102_0.5533_post_1x3x480x640.onnx
  }
] \
[-fdm {
    retinaface_mbn025_with_postprocess_Nx3x64x64_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x96x96_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x128x128_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x160x160_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x192x192_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x224x224_max001_th0.15.onnx,
    retinaface_mbn025_with_postprocess_Nx3x256x256_max001_th0.15.onnx,
    retinaface_resnet50_with_postprocess_Nx3x64x64_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x96x96_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x128x128_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x160x160_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x192x192_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x224x224_max001_th015.onnx,
    retinaface_resnet50_with_postprocess_Nx3x256x256_max001_th015.onnx,
  }
] \
[-nim NITEC_MODEL] \
[-v VIDEO] \
[-ep {cpu,cuda,tensorrt}] \
[-dvw]

options:
  -h, --help
    show this help message and exit
  -odm {...}, --object_detection_model {...}
    ONNX/TFLite file path for YOLOX.
  -fdm {...}, --face_detection_model {...}
    ONNX/TFLite file path for FaceDetection.
  -nim NITEC_MODEL, --nitec_model NITEC_MODEL
    ONNX/TFLite file path for NITEC.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic
    recording to MP4. Devices that use a MicroSD card or similar for main storage
    can speed up overall processing.
```

## 2. Acknowledgements

1. https://github.com/thohemp/nitec
