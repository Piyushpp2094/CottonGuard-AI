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

# Find the secondary leaf contour (area around 50,000 px, bbox around [50, 340, 270, 760])
sec_cnt = None
for c in cnts:
    area = cv2.contourArea(c)
    if 40000 < area < 100000:
        sec_cnt = c
        break

assert sec_cnt is not None, "Could not find secondary contour in UNet mask"
print(f"Found UNet secondary leaf contour: points={len(sec_cnt)}, area={cv2.contourArea(sec_cnt)}")

peri = cv2.arcLength(sec_cnt, True)
for eps in np.linspace(0.001, 0.01, 20):
    approx = cv2.approxPolyDP(sec_cnt, eps * peri, True)
    print(f"eps={eps:.4f}: verts={len(approx)}")
