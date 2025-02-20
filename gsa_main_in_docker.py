import argparse
import os
import cv2
import json
import torch
import time
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from gsa_image_acquirer_in_docker import ImageAcquirer
from gsa_image_saver_in_docker import ImageSaver
"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
parser.add_argument("--text-prompt", default="mouse. keyboard.")
parser.add_argument("--img-path", default="notebooks/images/truck.jpg")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--box-threshold", default="0.4")
parser.add_argument("--text-threshold", default="0.3")
# parser.add_argument("--output-dir", default="outputs/hf")
parser.add_argument("--no-dump-json", action="store_true")
parser.add_argument("--force-cpu", action="store_true")
parser.add_argument("--auto-save", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
TEXT_PROMPT = args.text_prompt
IMG_PATH = args.img_path
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
# OUTPUT_DIR = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
DUMP_JSON_RESULTS = not args.no_dump_json
AUTO_SAVE = args.auto_save
BOX_THRESHOLD = float(args.box_threshold)
TEXT_THRESHOLD = float(args.text_threshold)

# 0.创建环境
# create output directory
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 1.创建SAM2模型
# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# 2.创建GD模型
default_processor = "./AUTOPRECESSOR"
default_grounding_model = "./AUTOMODEL"
if os.path.exists(default_processor) and os.path.exists(default_grounding_model):
    print(f"Find processer and model locally.")
else:
    print(f"Didn't find {default_processor} and {default_grounding_model}, will download from hf. ")
    default_processor = default_grounding_model = GROUNDING_MODEL

processor = AutoProcessor.from_pretrained(default_processor)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(default_grounding_model).to(DEVICE)

if AUTO_SAVE:
    processor.save_pretrained("./AUTOPRECESSOR")
    grounding_model.save_pretrained("./AUTOMODEL")
    print(f"Model saved to local directory. {default_processor} and {default_grounding_model}")

# 3.loop， 获取图像，进行推理
image_acquirer = ImageAcquirer()
image_saver = ImageSaver()

text = TEXT_PROMPT
# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
while True:
    # 4.获得图像路径和oepncv imread格式的图片
    img_path, img_from_iacv = image_acquirer.getting_image() 
    if img_from_iacv is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img_cv_rgb = cv2.cvtColor(img_from_iacv, cv2.COLOR_BGR2RGB)
    image_acquirer.after_getting_image()

    start_time = time.time()

    sam2_predictor.set_image(img_cv_rgb)

    inputs = processor(images=img_cv_rgb, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # 修复尺寸获取方式
    height, width = img_cv_rgb.shape[:2]  # OpenCV的shape是 (H, W, C)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(height, width)]
    )

    """
    Results is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
            'labels': ['car', 'tire', 'tire', 'tire'], 
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """
    # breakpoint()
    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()  # array([], shape=(0, 4), dtype=float32)
    if len(input_boxes) == 0:
        print("WARN: No boxes detected, skipping SAM2 inference.")
        continue
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )


    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    end_time = time.time()
    print(f"fps: {1 / (end_time - start_time):.2f}")

    """
    Visualize image with supervision useful API
    """
    img = img_from_iacv
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)
    image_saver.save_image(annotated_frame)

    # """
    # Dump the results in standard format and save as json files
    # """

    # def single_mask_to_rle(mask):
    #     rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    #     rle["counts"] = rle["counts"].decode("utf-8")
    #     return rle

    # if DUMP_JSON_RESULTS:
    #     # convert mask into rle format
    #     mask_rles = [single_mask_to_rle(mask) for mask in masks]

    #     input_boxes = input_boxes.tolist()
    #     scores = scores.tolist()
    #     # save the results in standard format
    #     results = {
    #         "image_path": img_path,
    #         "annotations" : [
    #             {
    #                 "class_name": class_name,
    #                 "bbox": box,
    #                 "segmentation": mask_rle,
    #                 "score": score,
    #             }
    #             for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
    #         ],
    #         "box_format": "xyxy",
    #         "img_width": width,
    #         "img_height": height,
    #     }
        
    #     with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
    #         json.dump(results, f, indent=4)
