import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_h"](checkpoint="artifacts/models/sam_vit_h_4b8939.pth").to(device)
predictor = SamPredictor(sam)

with open("artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json", "r") as f:
    cvat_data = json.load(f)

# Frame 17
f17_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 17]
rects = [s for s in f17_shapes if s["id"] in [1206, 1207]]
img17 = cv2.imread("data/detection_consolidated_100/images/LHJ00116.jpg")
rgb17 = cv2.cvtColor(img17, cv2.COLOR_BGR2RGB)
predictor.set_image(rgb17)

for r in rects:
    box = np.array(r["points"])
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    best_idx = np.argmax(scores)
    mask = masks[best_idx].astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    print(f"Shape {r['id']}: vertices={len(pts)}, area={cv2.contourArea(approx):.1f}, box={box}")
