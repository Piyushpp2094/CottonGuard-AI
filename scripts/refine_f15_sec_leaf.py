import cv2, json, numpy as np, torch
from src.models.unet_resnet34 import build_unet_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_ckpt = "artifacts/checkpoints/segmentation/best_unet.pt"
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=unet_ckpt, device=device)
unet_model.eval()

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
norm = (rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
t = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
with torch.no_grad():
    out = unet_model(t)
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

leaf_mask = (pred >= 1).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
leaf_mask_closed = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
cnts, _ = cv2.findContours(leaf_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sec_cnt = next(c for c in cnts if 40000 < cv2.contourArea(c) < 100000)

# Rasterize UNet secondary contour
sec_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.drawContours(sec_mask, [sec_cnt], -1, 255, -1)

# Now include the chlorotic & necrotic lesions (1173 to 1179)
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f15_lesions = [c for c in cands if c['frame_number'] == 15 and c['replaced_shape_id'] in range(1173, 1180)]
for c in f15_lesions:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    cv2.fillPoly(sec_mask, [np.round(l_pts).astype(np.int32)], 255)

# Morphological closing to integrate lesions seamlessly into secondary leaf lamina
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
sec_mask_closed = cv2.morphologyEx(sec_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
sec_mask_closed = cv2.morphologyEx(sec_mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

cnts2, _ = cv2.findContours(sec_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest_sec = max(cnts2, key=cv2.contourArea)

peri = cv2.arcLength(largest_sec, True)
for eps in np.linspace(0.002, 0.006, 40):
    approx = cv2.approxPolyDP(largest_sec, eps * peri, True)
    if 20 <= len(approx) <= 35:
        break

sec_pts = approx.reshape(-1, 2).astype(np.float32)
sec_pts[:, 0] = np.clip(sec_pts[:, 0], 0.0, 800.0)
sec_pts[:, 1] = np.clip(sec_pts[:, 1], 0.0, 800.0)

print(f"CAND_F15_L_SECONDARY_01: verts={len(sec_pts)}, area={cv2.contourArea(sec_pts)}")

# Test containment of all 7 lesions
final_m15 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(final_m15, [np.round(sec_pts).astype(np.int32)], 1)

for c in f15_lesions:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_m, [np.round(l_pts).astype(np.int32)], 1)
    ov = np.sum((final_m15 == 1) & (l_m == 1))
    tot = np.sum(l_m == 1)
    print(f"  {c['candidate_id']}: {ov}/{tot} ({ov/tot*100:.1f}%)")
