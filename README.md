# NITEC-ONNX-TensorRT
ONNX implementation of "NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction" (switchable between RetinaFace or YOLOX)  https://github.com/thohemp/nitec

```
python demo_nitec_onnx_tflite.py -v 0
```

https://github.com/PINTO0309/NITEC-ONNX-TensorRT/assets/33194443/8f1bb9a1-11d7-4ac0-867e-e21241432c6e

```
usage: demo_nitec_onnx_tflite.py \
[-h]
[-odm {
  yolox_n_body_head_hand_post_0461_0.4428_1x3x480x640.onnx,
  yolox_t_body_head_hand_post_0461_0.4428_1x3x480x640.onnx,
  yolox_s_body_head_hand_post_0299_0.4983_1x3x480x640.onnx,
  yolox_m_body_head_hand_post_0299_0.5263_1x3x480x640.onnx,
  retinaface_mbn025_with_postprocess_480x640_max20_th0.70.onnx}
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
  -nim NITEC_MODEL, --nitec_model NITEC_MODEL
    ONNX/TFLite file path for NITEC.
  -v VIDEO, --video VIDEO
    Video file path or camera index.
  -ep {cpu,cuda,tensorrt}, --execution_provider {cpu,cuda,tensorrt}
    Execution provider for ONNXRuntime.
  -dvw, --disable_video_writer
    Disable video writer. Eliminates the file I/O load associated with automatic recording to MP4.
    Devices that use a MicroSD card or similar for main storage can speed up overall processing.
```
