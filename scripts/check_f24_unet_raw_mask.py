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

# Check pixel value at 1497 and 1499 centroids
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

for cid in ['CAND_F24_LES_1495', 'CAND_F24_LES_1497', 'CAND_F24_LES_1499']:
    c = next(x for x in cands if x['candidate_id'] == cid)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    overlap = np.sum((leaf_mask_closed == 255) & (l_mask == 1))
    tot = np.sum(l_mask == 1)
    print(f"{cid} overlap with raw UNet leaf_mask_closed: {overlap}/{tot} ({overlap/tot*100:.1f}%)")
