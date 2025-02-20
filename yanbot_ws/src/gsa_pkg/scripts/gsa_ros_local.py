#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 打印 Python 版本号（包含详细信息）
print("Python 版本:", sys.version)
print("版本元组:", sys.version_info)

# 获取 Python 解释器路径
print("解释器路径:", sys.executable)

# 获取标准库路径（Python 文件夹位置）
print("标准库路径:", os.path.dirname(sys.executable))

# 其他相关信息
print("操作系统:", sys.platform)


import rospy
import cv2
import numpy as np
from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision import sv
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from perception.grounded_sam2.gsa_image_acquirer_in_docker import ImageAcquirer
import json
import time
from pathlib import Path
import argparse

class GroundingDinoNode:
    def __init__(self):
        # ROS初始化
        rospy.init_node('grounding_dino_node', anonymous=True)
        
        # 参数设置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.force_cpu = rospy.get_param('~force_cpu', False)
        if self.force_cpu:
            self.device = "cpu"
        
        # 创建输出目录
        self.output_dir = Path(rospy.get_param('~output_dir', './outputs')) / time.strftime("%Y%m%d-%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型加载
        self.sam2_checkpoint = rospy.get_param('~sam2_checkpoint', './checkpoints/sam2.1_hiera_large.pt')
        self.sam2_model_config = rospy.get_param('~sam2_model_config', 'configs/sam2.1/sam2.1_hiera_l.yaml')
        self.grounding_model_name = rospy.get_param('~grounding_model', "IDEA-Research/grounding-dino-tiny")
        self.text_prompt = rospy.get_param('~text_prompt', "mouse. keyboard.")
        
        # 创建SAM2模型
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # 创建Grounding DINO模型
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_name).to(self.device)
        
        # 设置自动保存
        self.auto_save = rospy.get_param('~auto_save', True)
        if self.auto_save:
            self.processor.save_pretrained("./AUTOPRECESSOR")
            self.grounding_model.save_pretrained("./AUTOMODEL")
            rospy.loginfo("Models saved to local directory.")
        
        # ROS订阅者和发布者
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.result_pub = rospy.Publisher('/grounding_dino_results', String, queue_size=10)
        
        # 颜色映射
        self.color_palette = ColorPalette.from_hex(CUSTOM_COLOR_MAP)
        
        # 其他配置
        torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8 and self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
    def image_callback(self, msg):
        try:
            # 转换ROS图像消息为OpenCV格式
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        # 处理图像
        image = Image.fromarray(cv_image)
        image_np = np.array(image.convert("RGB"))
        
        # 设置SAM2输入
        self.sam2_predictor.set_image(image_np)
        
        # 处理Grounding DINO
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        
        # 处理SAM2掩模
        input_boxes = results[0]["boxes"].cpu().numpy()
        masks, scores, logits = self.sam2_predictor.predict(box=input_boxes, multimask_output=False)
        
        # 可视化
        detections = sv.Detections(xyxy=input_boxes, mask=masks.astype(bool), class_id=np.arange(len(results[0]["labels"])))
        
        # 创建标注图像
        annotated_frame = image.copy()
        box_annotator = sv.BoxAnnotator(color=self.color_palette)
        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        
        label_annotator = sv.LabelAnnotator(color=self.color_palette)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels=[f"{name} {score:.2f}" for name, score in zip(results[0]["labels"], scores)])
        
        mask_annotator = sv.MaskAnnotator(color=self.color_palette)
        final_image = mask_annotator.annotate(annotated_frame, detections)
        
        # 保存结果
        cv2.imwrite(self.output_dir / "annotated_image.jpg", final_image)
        
        # 转换为ROS图像消息并发布
        bridge = CvBridge()
        try:
            result_msg = bridge.cv2_to_imgmsg(final_image, encoding="bgr8")
            self.result_pub.publish(result_msg)
        except CvBridgeError as e:
            rospy.logerr(e)
        
        # 保存JSON结果
        if not rospy.get_param('~no_dump_json', False):
            self.save_json_results(results, image_np)
        
    def save_json_results(self, results, image_np):
        mask_rles = []
        for mask in masks:
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            mask_rles.append(rle)
        
        json_data = {
            "image_path": rospy.get_param('~img_path', ""),
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "score": score.item()
                }
                for class_name, box, mask_rle, score in zip(results[0]["labels"], input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": image_np.shape[1],
            "img_height": image_np.shape[0]
        }
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(json_data, f, indent=4)
        
if __name__ == "__main__":
    try:
        node = GroundingDinoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass