import torch, cv2, numpy as np
from pathlib import Path
from src.models.unet_resnet34 import build_unet_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_ckpt = "artifacts/checkpoints/segmentation/best_unet.pt"
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=unet_ckpt, device=device)
unet_model.eval()

frames = {
    15: "LHJ00001.jpg",
    21: "LR00140.jpg",
    24: "LR00578.jpg"
}

for f_idx, fn in frames.items():
    img_path = Path("data/detection_consolidated_100/images") / fn
    img = cv2.imread(str(img_path))
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
    
    print(f"\n--- Frame {f_idx} ({fn}): Found {len(cnts)} leaf contours from UNet ---")
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area > 1000:
            bx, by, bw, bh = cv2.boundingRect(c)
            print(f"  Contour {i}: area={area:.1f}, bbox=[{bx}, {by}, {bx+bw}, {by+bh}]")
