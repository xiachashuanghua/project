import numpy as np
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import time

cfg = LazyConfig.load("projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")#projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py"
# edit the config to utilize common Batch Norm
# cfg.model.backbone.bottom_up.stages.norm = "BN"
# cfg.model.backbone.norm = "BN"
model = instantiate(cfg.model)
model.cuda()
DetectionCheckpointer(model).load("/root/autodl-tmp/model_0016499.pth")  # load a file, usually from cfg.MODEL.WEIGHTS
# read image for inference input
# use PIL, to be consistent with evaluation#42, 87, 59, 34, 65, 23, 112, 7
img = torch.from_numpy(np.ascontiguousarray(read_image("/root/virus/val2017/0110.png", format="BGR")))
img = img.permute(2, 0, 1)  # HWC -> CHW
if torch.cuda.is_available():
    img = img.cuda()
inputs = [{"image": img}]
# run the model
model.eval()
with torch.no_grad():
    predictions_ls = model(inputs)
predictions = predictions_ls[0]

# Filter out predictions with confidence scores below 50%
scores = predictions["instances"].scores
high_confidence_predictions = predictions["instances"][scores >= 0.6]

# If no high confidence predictions, skip visualization
if len(high_confidence_predictions) == 0:
    print("No predictions with confidence >= 0.5, skipping visualization.")
else:
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    v = Visualizer(img_np, metadata=metadata)
    v = v.draw_instance_predictions(high_confidence_predictions.to("cpu"))
    cv2.imwrite("/root/demo/0001.png", v.get_image()[:, :, ::-1])