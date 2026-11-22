import cv2, json, numpy as np, torch
from src.models.unet_resnet34 import build_unet_resnet34
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_ckpt = "artifacts/checkpoints/segmentation/best_unet.pt"
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=unet_ckpt, device=device)
unet_model.eval()

# 1. Frame 15 Secondary Leaf
img15 = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
rgb15 = cv2.cvtColor(img15, cv2.COLOR_BGR2RGB)
norm15 = (rgb15 / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
t15 = torch.from_numpy(norm15.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
with torch.no_grad():
    out15 = unet_model(t15)
    pred15 = torch.argmax(out15, dim=1).squeeze().cpu().numpy()

leaf_m15 = (pred15 >= 1).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
leaf_m15_closed = cv2.morphologyEx(leaf_m15, cv2.MORPH_CLOSE, kernel)
cnts15, _ = cv2.findContours(leaf_m15_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sec_cnt15 = next(c for c in cnts15 if 40000 < cv2.contourArea(c) < 100000)

approx15 = cv2.approxPolyDP(sec_cnt15, 0.0062 * cv2.arcLength(sec_cnt15, True), True)
pts15 = approx15.reshape(-1, 2).astype(np.float32)
pts15[:, 0] = np.clip(pts15[:, 0], 0.0, 800.0)
pts15[:, 1] = np.clip(pts15[:, 1], 0.0, 800.0)
print(f"Frame 15 Secondary Leaf: verts={len(pts15)}, area={cv2.contourArea(pts15)}")

# 2. Frame 21 Top Leaf Blade
# In Frame 21, the top blade is located at y: [0, 60], x: [290, 640]
img21 = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv21 = cv2.cvtColor(img21, cv2.COLOR_BGR2HSV)
# Mask plant tissue in top stripe
mask21_top = np.zeros((800, 800), dtype=np.uint8)
# Get plant tissue by HSV
tissue21 = ((hsv21[:,:,0] >= 10) & (hsv21[:,:,0] <= 90) & (hsv21[:,:,1] >= 15)) | \
           (((hsv21[:,:,0] < 15) | (hsv21[:,:,0] > 165)) & (hsv21[:,:,1] >= 20))
mask21_top[0:70, 270:650] = tissue21[0:70, 270:650]
kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask21_top = cv2.morphologyEx(mask21_top, cv2.MORPH_CLOSE, kernel_sm, iterations=2)
cnts21, _ = cv2.findContours(mask21_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
top_cnt21 = max(cnts21, key=cv2.contourArea)
approx21 = cv2.approxPolyDP(top_cnt21, 0.005 * cv2.arcLength(top_cnt21, True), True)
pts21 = approx21.reshape(-1, 2).astype(np.float32)
pts21[:, 0] = np.clip(pts21[:, 0], 0.0, 800.0)
pts21[:, 1] = np.clip(pts21[:, 1], 0.0, 800.0)
print(f"Frame 21 Top Blade: verts={len(pts21)}, area={cv2.contourArea(pts21)}")

new_leaves = {
    "CAND_F15_L_SECONDARY_01": {
        "candidate_id": "CAND_F15_L_SECONDARY_01",
        "frame_number": 15,
        "annotation_label": "Leaf",
        "label_id": 5,
        "vertex_count": len(pts15),
        "area": float(cv2.contourArea(pts15)),
        "polygon_coordinates": pts15.flatten().tolist()
    },
    "CAND_F21_L_BLADE2": {
        "candidate_id": "CAND_F21_L_BLADE2",
        "frame_number": 21,
        "annotation_label": "Leaf",
        "label_id": 5,
        "vertex_count": len(pts21),
        "area": float(cv2.contourArea(pts21)),
        "polygon_coordinates": pts21.flatten().tolist()
    }
}

out_p = Path('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json')
out_p.parent.mkdir(parents=True, exist_ok=True)
with open(out_p, 'w') as f:
    json.dump(new_leaves, f, indent=2)

print("Saved new_secondary_leaf_candidates.json successfully.")
