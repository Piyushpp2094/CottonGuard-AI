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

# Check region [488:570, 57:91]
roi_pred = pred[57:91, 488:570]
print("Unique classes predicted in 1497 bbox:", np.unique(roi_pred, return_counts=True))
