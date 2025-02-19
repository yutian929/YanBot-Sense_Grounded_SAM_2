#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import json
import torch
import numpy as np
import rospy
import PIL
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class GroundedSAMNode:
    def __init__(self):
        rospy.init_node('grounded_sam_node', anonymous=True)
        
        # ROS parameters
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.text_prompt = rospy.get_param('~text_prompt', 'car. tire.')
        self.grounding_model_name = rospy.get_param('~grounding_model', "IDEA-Research/grounding-dino-tiny")
        self.sam2_checkpoint = rospy.get_param('~sam2_checkpoint', "checkpoints/sam2.1_hiera_large.pt")
        self.sam2_model_config = rospy.get_param('~sam2_model_config', "configs/sam2.1/sam2.1_hiera_l.yaml")
        self.force_cpu = rospy.get_param('~force_cpu', False)

        # Setup device
        self.device = "cuda" if torch.cuda.is_available() and not self.force_cpu else "cpu"
        rospy.loginfo(f"Using device: {self.device}")

        # Initialize models
        self._init_models()
        
        # ROS image bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.dino_pub = rospy.Publisher('groundingdino_annotated_image', Image, queue_size=1)
        self.sam_pub = rospy.Publisher('grounded_sam2_annotated_image_with_mask', Image, queue_size=1)
        
        # Subscriber
        self.image_sub = rospy.Subscriber(self.rgb_topic, Image, self.image_callback, queue_size=1)

    def _init_models(self):
        # Initialize SAM2
        rospy.loginfo("Initializing SAM2 model...")
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Initialize Grounding DINO
        rospy.loginfo("Initializing Grounding DINO model...")
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_name
        ).to(self.device)

    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            pil_image = PIL.Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            rospy.loginfo("OK-1")
            # Process image
            with torch.no_grad():
                # Run Grounding DINO
                inputs = self.processor(
                    images=pil_image, 
                    text=self.text_prompt, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.grounding_model(**inputs)
                
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[pil_image.size[::-1]]
                )

            # Get bounding boxes
            input_boxes = results[0]["boxes"].cpu().numpy()
            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            # Run SAM2
            self.sam2_predictor.set_image(np.array(pil_image))
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            # Prepare detections
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids
            )

            # Create annotations
            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(sv.CUSTOM_COLOR_MAP))
            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(sv.CUSTOM_COLOR_MAP))
            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(sv.CUSTOM_COLOR_MAP))

            # Annotate images
            annotated_frame = box_annotator.annotate(cv_image.copy(), detections)
            labels = [
                f"{cls} {conf:.2f}" for cls, conf in zip(class_names, confidences)
            ]
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            masked_frame = mask_annotator.annotate(annotated_frame.copy(), detections)

            # # Publish results
            # self._publish_image(annotated_frame, self.dino_pub)
            # self._publish_image(masked_frame, self.sam_pub)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def _publish_image(self, cv_image, publisher):
        try:
            msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            publisher.publish(msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {str(e)}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = GroundedSAMNode()
    node.run()