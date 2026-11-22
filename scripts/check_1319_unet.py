import cv2, json, numpy as np, torch
from src.models.unet_resnet34 import build_unet_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_ckpt = "artifacts/checkpoints/segmentation/best_unet.pt"
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=unet_ckpt, device=device)
unet_model.eval()

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
norm = (rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
t = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
with torch.no_grad():
    out = unet_model(t)
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

leaf_m = (pred >= 1).astype(np.uint8) * 255
# Check pixel values at 1319
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

c1319 = next(c for c in cands if c['candidate_id'] == 'CAND_F21_LES_1319')
pts = np.array(c1319['polygon_coordinates']).reshape(-1, 2)
m = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
ov = np.sum((leaf_m == 255) & (m == 1))
tot = np.sum(m == 1)
print(f"1319 overlap with raw UNet leaf_mask: {ov}/{tot} ({ov/tot*100:.1f}%)")
