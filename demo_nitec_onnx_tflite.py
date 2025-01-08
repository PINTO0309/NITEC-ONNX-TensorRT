#!/usr/bin/env python

"""
runtime: https://github.com/microsoft/onnxruntime

pip install onnxruntime or pip install onnxruntime-gpu
pip install opencv-contrib-python==4.9.0.80
"""
from __future__ import annotations
import os
import re
import sys
import copy
import cv2
import time
import requests
import subprocess
import numpy as np
from enum import Enum
from pprint import pprint
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, Optional, List, Dict
import importlib.util
from abc import ABC, abstractmethod

# https://developer.nvidia.com/cuda-gpus
NVIDIA_GPU_MODELS_CC = [
    'RTX 3050', 'RTX 3060', 'RTX 3070', 'RTX 3080', 'RTX 3090',
]

ONNX_TRTENGINE_SETS = {
    'yolov9_n_wholebody25_post_0100_1x3x480x640.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_2697494743358455023_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_2697494743358455023_0_0_fp16_sm86.profile',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_2697494743358455023_1_1_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_2697494743358455023_1_1_fp16_sm86.profile',
    ],
    'nitec_rs18_e20_Nx3x224x224.onnx': [
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_7593398840732104190_0_0_fp16_sm86.engine',
        'TensorrtExecutionProvider_TRTKernel_graph_main_graph_7593398840732104190_0_0_fp16_sm86.profile',
    ],
}

class Color(Enum):
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERSE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

    def __str__(self):
        return self.value

    def __call__(self, s):
        return str(self) + str(s) + str(Color.RESET)

@dataclass(frozen=False)
class Box():
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    looked_score: float
    generation: int = -1 # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1 # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1 # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1 # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False

class AbstractModel(ABC):
    """AbstractModel
    Base class of the model.
    """
    _runtime: str = 'onnx'
    _model_path: str = ''
    _obj_class_score_th: float = 0.35
    _attr_class_score_th: float = 0.70
    _input_shapes: List[List[int]] = []
    _input_names: List[str] = []
    _output_shapes: List[List[int]] = []
    _output_names: List[str] = []

    # onnx/tflite
    _interpreter = None
    _inference_model = None
    _providers = None
    _swap = (2, 0, 1)
    _h_index = 2
    _w_index = 3

    # onnx
    _onnx_dtypes_to_np_dtypes = {
        "tensor(float)": np.float32,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
    }

    # tflite
    _input_details = None
    _output_details = None

    @abstractmethod
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = '',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self._runtime = runtime
        self._model_path = model_path
        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._providers = providers

        # Model loading
        if self._runtime == 'onnx':
            import onnxruntime # type: ignore
            onnxruntime.set_default_logger_severity(3) # ERROR
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            self._interpreter = \
                onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_option,
                    providers=providers,
                )
            self._providers = self._interpreter.get_providers()
            print(f'{Color.GREEN("Enabled ONNX ExecutionProviders:")}')
            pprint(f'{self._providers}')

            import onnx
            onnx_graph: onnx.ModelProto = onnx.load(model_path)
            if onnx_graph.graph.node[0].op_type == "Resize":
                first_resize_op: List[onnx.ValueInfoProto] = [i for i in onnx_graph.graph.value_info if i.name == "prep/Resize_output_0"]
                if first_resize_op:
                    self._input_shapes = [[d.dim_value for d in first_resize_op[0].type.tensor_type.shape.dim]]
                else:
                    self._input_shapes = [
                        input.shape for input in self._interpreter.get_inputs()
                    ]
            else:
                self._input_shapes = [
                    input.shape for input in self._interpreter.get_inputs()
                ]


            self._input_names = [
                input.name for input in self._interpreter.get_inputs()
            ]
            self._input_dtypes = [
                self._onnx_dtypes_to_np_dtypes[input.type] for input in self._interpreter.get_inputs()
            ]
            self._output_shapes = [
                output.shape for output in self._interpreter.get_outputs()
            ]
            self._output_names = [
                output.name for output in self._interpreter.get_outputs()
            ]
            self._model = self._interpreter.run
            self._swap = (2, 0, 1)
            self._h_index = 2
            self._w_index = 3

        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            if self._runtime == 'tflite_runtime':
                from tflite_runtime.interpreter import Interpreter # type: ignore
                self._interpreter = Interpreter(model_path=model_path)
            elif self._runtime == 'tensorflow':
                import tensorflow as tf # type: ignore
                self._interpreter = tf.lite.Interpreter(model_path=model_path)
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            self._input_names = [
                input.get('name', None) for input in self._input_details
            ]
            self._input_dtypes = [
                input.get('dtype', None) for input in self._input_details
            ]
            self._output_shapes = [
                output.get('shape', None) for output in self._output_details
            ]
            self._output_names = [
                output.get('name', None) for output in self._output_details
            ]
            self._model = self._interpreter.get_signature_runner()
            self._swap = (0, 1, 2)
            self._h_index = 1
            self._w_index = 2

    @abstractmethod
    def __call__(
        self,
        *,
        input_datas: List[np.ndarray],
    ) -> List[np.ndarray]:
        datas = {
            f'{input_name}': input_data \
                for input_name, input_data in zip(self._input_names, input_datas)
        }
        if self._runtime == 'onnx':
            outputs = [
                output for output in \
                    self._model(
                        output_names=self._output_names,
                        input_feed=datas,
                    )
            ]
            return outputs
        elif self._runtime in ['tflite_runtime', 'tensorflow']:
            outputs = [
                output for output in \
                    self._model(
                        **datas
                    ).values()
            ]
            return outputs

    @abstractmethod
    def _preprocess(
        self,
        *,
        image: np.ndarray,
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess(
        self,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> List[Box]:
        raise NotImplementedError()

class YOLOv9(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'yolov9_n_wholebody25_post_0100_1x3x480x640.onnx',
        obj_class_score_th: Optional[float] = 0.35,
        attr_class_score_th: Optional[float] = 0.70,
        providers: Optional[List] = None,
    ):
        """

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for YOLOv9. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for YOLOv9

        obj_class_score_th: Optional[float]
            Object score threshold. Default: 0.35

        attr_class_score_th: Optional[float]
            Attributes score threshold. Default: 0.70

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            obj_class_score_th=obj_class_score_th,
            attr_class_score_th=attr_class_score_th,
            providers=providers,
        )
        self.mean: np.ndarray = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9
        self.std: np.ndarray = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape([3,1,1]) # Not used in YOLOv9

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """

        Parameters
        ----------
        image: np.ndarray
            Entire image

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, atrributes, is_used=False]
        """
        temp_image = copy.deepcopy(image)
        # PreProcess
        resized_image = \
            self._preprocess(
                temp_image,
            )
        # Inference
        inferece_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = super().__call__(input_datas=[inferece_image])
        boxes = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                image=temp_image,
                boxes=boxes,
                disable_generation_identification_mode=disable_generation_identification_mode,
                disable_gender_identification_mode=disable_gender_identification_mode,
                disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
                disable_headpose_identification_mode=disable_headpose_identification_mode,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        image = image.transpose(self._swap)
        image = \
            np.ascontiguousarray(
                image,
                dtype=np.float32,
            )
        return image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image.

        boxes: np.ndarray
            float32[N, 7]. [instances, [batchno, classid, score, x1, y1, x2, y2]].

        disable_generation_identification_mode: bool

        disable_gender_identification_mode: bool

        disable_left_and_right_hand_identification_mode: bool

        disable_headpose_identification_mode: bool

        Returns
        -------
        result_boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2, cx, cy, attributes, is_used=False]
        """
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        if len(boxes) > 0:
            scores = boxes[:, 2:3]
            keep_idxs = scores[:, 0] > self._obj_class_score_th
            scores_keep = scores[keep_idxs, :]
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                # Object filter
                for box, score in zip(boxes_keep, scores_keep):
                    classid = int(box[1])
                    x_min = int(max(0, box[3]) * image_width / self._input_shapes[0][self._w_index])
                    y_min = int(max(0, box[4]) * image_height / self._input_shapes[0][self._h_index])
                    x_max = int(min(box[5], self._input_shapes[0][self._w_index]) * image_width / self._input_shapes[0][self._w_index])
                    y_max = int(min(box[6], self._input_shapes[0][self._h_index]) * image_height / self._input_shapes[0][self._h_index])
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=float(score),
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                            looked_score=0.0,
                            generation=-1, # -1: Unknown, 0: Adult, 1: Child
                            gender=-1, # -1: Unknown, 0: Male, 1: Female
                            handedness=-1, # -1: Unknown, 0: Left, 1: Right
                            head_pose=-1, # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
                        )
                    )
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes \
                        if (box.classid in [1,2,3,4,8,9,10,11,12,13,14,15] and box.score >= self._attr_class_score_th) or box.classid not in [1,2,3,4,8,9,10,11,12,13,14,15]
                ]

                # Adult, Child merge
                # classid: 0 -> Body
                #   classid: 1 -> Adult
                #   classid: 2 -> Child
                # 1. Calculate Adult and Child IoUs for Body detection results
                # 2. Connect either the Adult or the Child with the highest score and the highest IoU with the Body.
                # 3. Exclude Adult and Child from detection results
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]
                # Male, Female merge
                # classid: 0 -> Body
                #   classid: 3 -> Male
                #   classid: 4 -> Female
                # 1. Calculate Male and Female IoUs for Body detection results
                # 2. Connect either the Male or the Female with the highest score and the highest IoU with the Body.
                # 3. Exclude Male and Female from detection results
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]
                # HeadPose merge
                # classid: 7 -> Head
                #   classid:  8 -> Front
                #   classid:  9 -> Right-Front
                #   classid: 10 -> Right-Side
                #   classid: 11 -> Right-Back
                #   classid: 12 -> Back
                #   classid: 13 -> Left-Back
                #   classid: 14 -> Left-Side
                #   classid: 15 -> Left-Front
                # 1. Calculate HeadPose IoUs for Head detection results
                # 2. Connect either the HeadPose with the highest score and the highest IoU with the Head.
                # 3. Exclude HeadPose from detection results
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8,9,10,11,12,13,14,15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8,9,10,11,12,13,14,15]]
                # Left and right hand merge
                # classid: 21 -> Hand
                #   classid: 22 -> Left-Hand
                #   classid: 23 -> Right-Hand
                # 1. Calculate Left-Hand and Right-Hand IoUs for Hand detection results
                # 2. Connect either the Left-Hand or the Right-Hand with the highest score and the highest IoU with the Hand.
                # 3. Exclude Left-Hand and Right-Hand from detection results
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 21]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [22, 23]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [22, 23]]
        return result_boxes

    def _find_most_relevant_obj(
        self,
        *,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                # Process only unused objects with center Euclidean distance less than or equal to 10.0
                if not target_obj.is_used and distance <= 10.0:
                    # Prioritize high-score objects
                    if target_obj.score >= best_score:
                        # IoU Calculation
                        iou: float = \
                            self._calculate_iou(
                                base_obj=base_obj,
                                target_obj=target_obj,
                            )
                        # Adopt object with highest IoU
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            # Calculate the Euclidean distance between the center coordinates
                            # of the base and the center coordinates of the target
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score
            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True

                elif most_relevant_obj.classid == 22:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 23:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _calculate_iou(
        self,
        *,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        # Calculate areas of overlap
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)
        # If there is no overlap
        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0
        # Calculate area of overlap and area of each bounding box
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)
        # Calculate IoU
        iou = inter_area / float(area1 + area2 - inter_area)
        return iou

class NITEC(AbstractModel):
    def __init__(
        self,
        *,
        runtime: Optional[str] = 'onnx',
        model_path: Optional[str] = 'nitec_rs18_e20_Nx3x224x224.onnx',
        providers: Optional[List] = None,
    ):
        """NITEC

        Parameters
        ----------
        runtime: Optional[str]
            Runtime for NITEC. Default: onnx

        model_path: Optional[str]
            ONNX/TFLite file path for NITEC

        providers: Optional[List]
            Providers for ONNXRuntime.
        """
        super().__init__(
            runtime=runtime,
            model_path=model_path,
            providers=providers,
        )
        self.swap = (0,3,1,2)
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,3,1,1)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,3,1,1)

    def __call__(
        self,
        image: np.ndarray,
        boxes: List[Box],
    ) -> List[Box]:
        """NITEC

        Parameters
        ----------
        image: np.ndarray
            Entire image

        boxes: List[Box]
            Face boxes

        Returns
        -------
        boxes: List[Box]
            List of Boxes with a set score looking at the camera for each face.
        """
        temp_image = copy.deepcopy(image)
        temp_boxes = copy.deepcopy(boxes)
        # PreProcess
        face_images = \
            self._preprocess(
                image=temp_image,
                boxes=temp_boxes,
            )
        # Inference
        outputs = super().__call__(input_datas=[face_images])
        looked_scores = outputs[0]
        # PostProcess
        result_boxes = \
            self._postprocess(
                looked_scores=looked_scores,
                boxes=temp_boxes,
            )
        return result_boxes

    def _preprocess(
        self,
        image: np.ndarray,
        boxes: List[Box],
    ) -> np.ndarray:
        """_preprocess

        Parameters
        ----------
        image: np.ndarray
            Entire image

        boxes: List[Box]
            Face boxes

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        resized_image: np.ndarray
            Resized and normalized image.
        """
        # Resize + Transpose
        resized_images: List[np.ndarray] = []
        for box in boxes:
            croped_image = image[box.y1:box.y2, box.x1:box.x2, :]
            resized_image = cv2.resize(
                croped_image,
                (
                    int(self._input_shapes[0][self._w_index]),
                    int(self._input_shapes[0][self._h_index]),
                )
            )
            resized_images.append(resized_image)
        if len(resized_images) == 0:
            resized_images = np.zeros([0, self._input_shapes[0][self._h_index], self._input_shapes[0][self._w_index], 3], dtype=np.float32)
        face_images = np.asarray(resized_images, dtype=self._input_dtypes[0])
        face_images = face_images[..., ::-1] # BGR to RGB
        face_images = face_images.transpose(self.swap)
        face_images = (face_images / 255.0 - self.mean) / self.std
        return face_images

    def _postprocess(
        self,
        looked_scores: np.ndarray,
        boxes: List[Box],
    ) -> List[Box]:
        """_postprocess

        Parameters
        ----------
        looked_scores: np.ndarray
            Score looking at the camera face by face.

        boxes: List[Box]
            List of bounding boxes detected by object detection.

        Returns
        -------
        boxes: List[Box]
            Predicted boxes: [classid, score, x1, y1, x2, y2]
        """
        for box, looked_score in zip(boxes, looked_scores):
            box.looked_score = float(looked_score)
        return boxes

def is_parsable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_package_installed(package_name: str):
    """Checks if the specified package is installed.

    Parameters
    ----------
    package_name: str
        Name of the package to be checked.

    Returns
    -------
    result: bool
        True if the package is installed, false otherwise.
    """
    return importlib.util.find_spec(package_name) is not None

def download_file(url, folder, filename):
    """
    Download a file from a URL and save it to a specified folder.
    If the folder does not exist, it is created.

    :param url: URL of the file to download.
    :param folder: Folder where the file will be saved.
    :param filename: Filename to save the file.
    """
    # Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Full path for the file
    file_path = os.path.join(folder, filename)
    # Download the file
    print(f"{Color.GREEN('Downloading...')} {url} to {file_path}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"{Color.GREEN('Download completed:')} {file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def get_nvidia_gpu_model() -> List[str]:
    try:
        # Run nvidia-smi command
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)

        # Extract GPU model numbers using regular expressions
        models = re.findall(r'GPU \d+: (.*?)(?= \(UUID)', output)
        return models
    except Exception as e:
        print(f"Error: {e}")
        return []

def draw_dashed_line(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Function to draw a dashed line"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(image, tuple(start), tuple(end), color, thickness)

def draw_dashed_rectangle(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10
):
    """Function to draw a dashed rectangle"""
    tl_tr = (bottom_right[0], top_left[1])
    bl_br = (top_left[0], bottom_right[1])
    draw_dashed_line(image, top_left, tl_tr, color, thickness, dash_length)
    draw_dashed_line(image, tl_tr, bottom_right, color, thickness, dash_length)
    draw_dashed_line(image, bottom_right, bl_br, color, thickness, dash_length)
    draw_dashed_line(image, bl_br, top_left, color, thickness, dash_length)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-odm',
        '--object_detection_model',
        type=str,
        default='yolov9_n_wholebody25_post_0100_1x3x480x640.onnx',
        choices=[
            'yolov9_n_wholebody25_post_0100_1x3x480x640.onnx',
        ],
        help='ONNX/TFLite file path for YOLOX.',
    )
    parser.add_argument(
        '-nim',
        '--nitec_model',
        type=str,
        default='nitec_rs18_e20_Nx3x224x224.onnx',
        help='ONNX/TFLite file path for NITEC.',
    )
    parser.add_argument(
        '-v',
        '--video',
        type=str,
        default="0",
        help='Video file path or camera index.',
    )
    parser.add_argument(
        '-ep',
        '--execution_provider',
        type=str,
        choices=['cpu', 'cuda', 'tensorrt'],
        default='tensorrt',
        help='Execution provider for ONNXRuntime.',
    )
    parser.add_argument(
        '-ost',
        '--object_socre_threshold',
        type=float,
        default=0.35,
        help=\
            'The detection score threshold for object detection. Default: 0.35',
    )
    parser.add_argument(
        '-ast',
        '--attribute_socre_threshold',
        type=float,
        default=0.75,
        help=\
            'The attribute score threshold for object detection. Default: 0.70',
    )
    parser.add_argument(
        '-dvw',
        '--disable_video_writer',
        action='store_true',
        help=\
            'Disable video writer. '+
            'Eliminates the file I/O load associated with automatic recording to MP4. '+
            'Devices that use a MicroSD card or similar for main storage can speed up overall processing.',
    )
    args = parser.parse_args()

    # runtime check
    object_detection_model_file: str = args.object_detection_model
    nitec_model_file: str = args.nitec_model
    object_socre_threshold: float = args.object_socre_threshold
    attribute_socre_threshold: float = args.attribute_socre_threshold
    object_detection_model_ext: str = os.path.splitext(object_detection_model_file)[1][1:].lower()
    nitec_model_file_ext: str = os.path.splitext(nitec_model_file)[1][1:].lower()
    runtime: str = None
    if object_detection_model_ext != nitec_model_file_ext:
        print(Color.RED('ERROR: object_detection_model and feature_extractor_model must be files with the same extension.'))
        sys.exit(0)
    if object_detection_model_ext == 'onnx':
        if not is_package_installed('onnxruntime'):
            print(Color.RED('ERROR: onnxruntime is not installed. pip install onnxruntime or pip install onnxruntime-gpu'))
            sys.exit(0)
        runtime = 'onnx'
    elif object_detection_model_ext == 'tflite':
        if is_package_installed('tflite_runtime'):
            runtime = 'tflite_runtime'
        elif is_package_installed('tensorflow'):
            runtime = 'tensorflow'
        else:
            print(Color.RED('ERROR: tflite_runtime or tensorflow is not installed.'))
            print(Color.RED('ERROR: https://github.com/PINTO0309/TensorflowLite-bin'))
            print(Color.RED('ERROR: https://github.com/tensorflow/tensorflow'))
            sys.exit(0)

    WEIGHT_FOLDER_PATH = '.'
    gpu_models = get_nvidia_gpu_model()
    default_supported_gpu_model = False
    if len(gpu_models) == 1:
        gpu_model = gpu_models[0]
        for target_gpu_model in NVIDIA_GPU_MODELS_CC:
            if target_gpu_model in gpu_model:
                default_supported_gpu_model = True
                break

    # Download object detection onnx
    weight_file = os.path.basename(object_detection_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/cuda-12.5-tensorrt-10.6.0/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download YOLOv9 tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/cuda-12.5-tensorrt-10.6.0/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    # Download NITEC onnx
    weight_file = os.path.basename(nitec_model_file)
    if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, weight_file)):
        url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/cuda-12.5-tensorrt-10.6.0/{weight_file}"
        download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=weight_file)
    # Download NITEC tensorrt engine
    if default_supported_gpu_model:
        trt_engine_files = ONNX_TRTENGINE_SETS.get(weight_file, None)
        if trt_engine_files is not None:
            for trt_engine_file in trt_engine_files:
                if not os.path.isfile(os.path.join(WEIGHT_FOLDER_PATH, trt_engine_file)):
                    url = f"https://github.com/PINTO0309/NITEC-ONNX-TensorRT/releases/download/cuda-12.5-tensorrt-10.6.0/{trt_engine_file}"
                    download_file(url=url, folder=WEIGHT_FOLDER_PATH, filename=trt_engine_file)

    video: str = args.video
    execution_provider: str = args.execution_provider
    providers: List[Tuple[str, Dict] | str] = None
    if execution_provider == 'cpu':
        providers = [
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'cuda':
        providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]
    elif execution_provider == 'tensorrt':
        providers = [
            (
                'TensorrtExecutionProvider', {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': '.',
                    'trt_fp16_enable': True,
                }
            ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ]

    # Model initialization
    object_detection_model = \
        YOLOv9(
            runtime=runtime,
            model_path=object_detection_model_file,
            obj_class_score_th=object_socre_threshold,
            attr_class_score_th=attribute_socre_threshold,
            providers=providers,
        )
    nitec_model = \
        NITEC(
            runtime=runtime,
            model_path=nitec_model_file,
            providers=providers,
        )

    cap = cv2.VideoCapture(
        int(video) if is_parsable_to_int(video) else video
    )
    disable_video_writer: bool = args.disable_video_writer
    video_writer = None
    if not disable_video_writer:
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            filename='output.mp4',
            fourcc=fourcc,
            fps=cap_fps,
            frameSize=(w, h),
        )

    while cap.isOpened():
        res, image = cap.read()
        if not res:
            break

        debug_image = copy.deepcopy(image)
        # debug_image_h = debug_image.shape[0]
        debug_image_w = debug_image.shape[1]

        start_time = time.perf_counter()

        boxes = object_detection_model(
            image=debug_image,
            disable_generation_identification_mode=True,
            disable_gender_identification_mode=True,
            disable_left_and_right_hand_identification_mode=True,
            disable_headpose_identification_mode=True,
        )
        face_boxes = \
            [box for box in boxes if box.classid == 16] if isinstance(object_detection_model, YOLOv9) else [box for box in boxes if box.classid == 1]
        other_boxes = \
            [box for box in boxes if box.classid != 16] if isinstance(object_detection_model, YOLOv9) else [box for box in boxes if box.classid != 0]
        looked_or_not_face_boxes: List[Box] = []
        if len(face_boxes) > 0:
            if len(face_boxes) > 0:
                looked_or_not_face_boxes = nitec_model(image=debug_image, boxes=face_boxes)

        elapsed_time = time.perf_counter() - start_time
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{elapsed_time*1000:.2f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        for box in looked_or_not_face_boxes:
            draw_dashed_rectangle(
                image=debug_image,
                top_left=(box.x1, box.y1),
                bottom_right=(box.x2, box.y2),
                color=(255, 255, 255),
                thickness=2,
                dash_length=10,
            )
            draw_dashed_rectangle(
                image=debug_image,
                top_left=(box.x1, box.y1),
                bottom_right=(box.x2, box.y2),
                color=(0, 200, 255),
                thickness=1,
                dash_length=10,
            )
            ptx = box.x1 if box.x1+50 < debug_image_w else debug_image_w-50
            pty = box.y1-10 if box.y1-25 > 0 else 20
            if box.looked_score >= 0.15:
                cv2.putText(debug_image, 'Looked', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(debug_image, 'Looked', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break

        cv2.imshow("test", debug_image)
        if video_writer is not None:
            video_writer.write(debug_image)

    if video_writer is not None:
        video_writer.release()

    if cap is not None:
        cap.release()


if __name__ == "__main__":
    main()
