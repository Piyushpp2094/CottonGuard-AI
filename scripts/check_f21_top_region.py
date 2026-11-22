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

# Inspect top region y < 100
top_pred = pred[0:100, :]
print("Top region pred classes counts:", np.unique(top_pred, return_counts=True))

# Check lesions 1341-1344 bboxes
with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

for rep_id in [1341, 1342, 1343, 1344]:
    c = next(x for x in master_cands if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    print(f"Lesion {rep_id}: bbox=[{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}]")
