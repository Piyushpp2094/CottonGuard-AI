import cv2, json, numpy as np, torch
from src.models.unet_resnet34 import build_unet_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_ckpt = "artifacts/checkpoints/segmentation/best_unet.pt"
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=unet_ckpt, device=device)
unet_model.eval()

img = cv2.imread("data/detection_consolidated_100/images/LR00578.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
norm = (rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
t = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
with torch.no_grad():
    out = unet_model(t)
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()

leaf_mask = (pred >= 1).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
leaf_mask_closed = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

cnts, _ = cv2.findContours(leaf_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(cnts, key=cv2.contourArea)

# Check points of largest where x > 400 and y < 150
raw_pts = largest.reshape(-1, 2)
apical_pts = raw_pts[(raw_pts[:, 0] > 400) & (raw_pts[:, 1] < 150)]
print("Raw contour points in apical region (x > 400, y < 150):")
print(apical_pts)
