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

raw_leaf_m = (pred >= 1).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
raw_leaf_closed = cv2.morphologyEx(raw_leaf_m, cv2.MORPH_CLOSE, kernel)

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

for rep_id in [1327, 1334, 1337]:
    c = next(x for x in master_cands if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_m, [np.round(pts).astype(np.int32)], 1)
    tot = np.sum(l_m == 1)
    ov = np.sum((raw_leaf_closed == 255) & (l_m == 1))
    print(f"Rep {rep_id} ({c['candidate_id']}) overlap with raw UNet leaf: {ov} / {tot} ({ov/tot*100:.2f}%)")
