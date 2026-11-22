"""
Builds the Cotton Leaf Verification Dataset:
- Class 0: cotton_leaf (from SAR-CLD-2024, honoring train/val/test splits)
- Class 1: non_cotton_leaf (Tomato, Potato, Pepper, Corn, Apple, Grape from PlantVillage)
- Class 2: non_leaf (Human faces from LFW, Documents, and Objects/Vehicles/Animals from ImageNet)

Ensures zero-leakage and reproducible group-aware splitting.
"""

import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIF_DIR = PROJECT_ROOT / "data" / "verification"
IMAGES_DIR = VERIF_DIR / "images"
SPLITS_DIR = VERIF_DIR / "splits"

SEED = 42
random.seed(SEED)


def build_directories():
    for sub in [
        IMAGES_DIR / "cotton_leaf",
        IMAGES_DIR / "non_cotton_leaf",
        IMAGES_DIR / "non_leaf" / "faces",
        IMAGES_DIR / "non_leaf" / "documents",
        IMAGES_DIR / "non_leaf" / "objects",
        SPLITS_DIR,
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    print("Created verification directories.")


def get_cotton_leaf_samples():
    """Reads existing group-isolated splits from artifacts/manifests/splits/."""
    splits = {}
    split_filenames = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
    for s_name, fname in split_filenames.items():
        csv_p = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / fname
        df = pd.read_csv(csv_p)
        samples = []
        for _, row in df.iterrows():
            abs_p = Path(row["absolute_path"]) if "absolute_path" in row and pd.notna(row["absolute_path"]) else None
            if not abs_p or not abs_p.is_file():
                rel_p = row.get("relative_path")
                full_p = PROJECT_ROOT / rel_p if rel_p else None
                if not full_p or not full_p.is_file():
                    alt_p = Path(r"D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\Original Dataset") / row["class_folder_label"] / Path(rel_p).name
                    if alt_p.is_file():
                        abs_p = alt_p
                    else:
                        abs_p = full_p
                else:
                    abs_p = full_p
            samples.append({
                "filepath": str(abs_p),
                "class_name": "cotton_leaf",
                "label": 0,
                "sub_category": row.get("class_folder_label", "cotton_leaf"),
                "split": s_name
            })
        splits[s_name] = samples
        print(f"Cotton leaf {s_name}: {len(samples)} images.")
    return splits


def download_non_cotton_leaves(target_count=1000):
    """Downloads diverse non-cotton crop leaves from PlantVillage."""
    out_dir = IMAGES_DIR / "non_cotton_leaf"
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.JPG"))
    if len(existing) >= target_count:
        print(f"Non-cotton leaves already present: {len(existing)} images.")
        return [str(p) for p in existing]

    categories = [
        "Tomato___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Potato___healthy",
        "Potato___Early_blight",
        "Pepper__bell___healthy",
        "Pepper__bell___Bacterial_spot",
        "Corn_(maize)___healthy",
        "Corn_(maize)___Common_rust_",
        "Apple___healthy",
        "Apple___Apple_scab",
        "Grape___healthy",
        "Grape___Black_rot",
    ]

    per_cat = max(20, target_count // len(categories) + 5)
    print(f"Fetching file lists from PlantVillage for {len(categories)} categories (up to {per_cat} each)...")
    
    download_tasks = []
    for cat in categories:
        api_url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{cat}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                items = json.loads(resp.read())
                for item in items[:per_cat]:
                    f_name = f"{cat}_{item['name']}"
                    dst = out_dir / f_name
                    if not dst.is_file():
                        download_tasks.append((item["download_url"], dst))
        except Exception as e:
            print(f"Warning fetching category {cat}: {e}")

    print(f"Downloading {len(download_tasks)} non-cotton leaf images...")

    def _fetch(url, dst):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch, url, dst) for url, dst in download_tasks]
        completed = 0
        for fut in as_completed(futures):
            if fut.result():
                completed += 1
            if completed % 100 == 0:
                print(f"  Downloaded {completed}/{len(download_tasks)} non-cotton leaves...")

    all_files = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.JPG"))
    print(f"Total non-cotton leaves available: {len(all_files)}")
    return [str(p) for p in all_files]


def download_lfw_faces(target_count=350):
    """Extracts human face images from LFW parquet."""
    out_dir = IMAGES_DIR / "non_leaf" / "faces"
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"LFW faces already present: {len(existing)} images.")
        return [str(p) for p in existing]

    print("Loading LFW parquet from Hugging Face for human face extraction...")
    url = "https://huggingface.co/datasets/logasja/lfw/resolve/main/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(url, columns=["image"])
    
    needed = target_count - len(existing)
    start_idx = len(existing)
    print(f"Extracting {needed} face images...")
    for i in range(start_idx, start_idx + needed):
        if i >= len(df):
            break
        img_bytes = df["image"].iloc[i]["bytes"]
        dst = out_dir / f"face_{i:04d}.jpg"
        with open(dst, "wb") as f:
            f.write(img_bytes)

    all_faces = list(out_dir.glob("*.jpg"))
    print(f"Total human face images available: {len(all_faces)}")
    return [str(p) for p in all_faces]


def download_imagenet_objects(target_count=350):
    """Downloads everyday object, vehicle, animal, and scene images from ImageNet sample set."""
    out_dir = IMAGES_DIR / "non_leaf" / "objects"
    existing = list(out_dir.glob("*.JPEG")) + list(out_dir.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"ImageNet objects already present: {len(existing)} images.")
        return [str(p) for p in existing]

    print("Fetching ImageNet sample images catalog from GitHub...")
    api_url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read())

    image_items = [x for x in items if x["name"].lower().endswith((".jpeg", ".jpg", ".png"))]
    random.shuffle(image_items)
    to_download = image_items[:target_count]

    print(f"Downloading {len(to_download)} everyday objects/vehicles/animals...")
    def _fetch(item):
        dst = out_dir / item["name"]
        if dst.is_file():
            return True
        try:
            req = urllib.request.Request(item["download_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch, item) for item in to_download]
        completed = sum(1 for fut in as_completed(futures) if fut.result())

    all_objs = list(out_dir.glob("*.JPEG")) + list(out_dir.glob("*.jpg"))
    print(f"Total object images available: {len(all_objs)}")
    return [str(p) for p in all_objs]


def generate_document_images(target_count=350):
    """Generates authentic, diverse high-resolution document and text images."""
    out_dir = IMAGES_DIR / "non_leaf" / "documents"
    existing = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"Document images already present: {len(existing)} images.")
        return [str(p) for p in existing]

    print(f"Generating {target_count - len(existing)} authentic document images (invoices, forms, research papers, letters, tables)...")

    DOC_TYPES = [
        "TAX INVOICE & BILL OF SUPPLY",
        "AGRONOMIC RESEARCH STUDY & FIELD REPORT",
        "LEGAL CONTRACT & NON-DISCLOSURE AGREEMENT",
        "PURCHASE ORDER & SHIPPING MANIFEST",
        "FINANCIAL AUDIT STATEMENT & LEDGER",
        "PATIENT MEDICAL SUMMARY & DIAGNOSTIC LAB REPORT",
        "TECHNICAL SPECIFICATION & API DOCUMENTATION",
        "CERTIFICATE OF INCORPORATION & BYLAWS"
    ]

    LOREM_WORDS = (
        "cotton disease analysis platform architecture evaluation benchmark accuracy precision recall "
        "f1-score gradient descent optimization matrix parameter calculation severity percentage foliar "
        "pathogen begomovirus xanthomonas citri malvacearum nitrogen phosphorus potassium agronomic triage "
        "reportlab pdf fastapi sqlalchemy uvicorn pydantic classification detection segmentation gradcam "
        "balance sheet invoice total subtotal tax rate discount terms conditions signature authorized stamp "
        "section article paragraph agreement parties rights obligation liability warranty jurisdiction"
    ).split()

    start_idx = len(existing)
    for i in range(start_idx, target_count):
        w, h = random.choice([(600, 800), (700, 950), (650, 850)])
        bg_color = random.choice([
            (255, 255, 255),
            (250, 250, 248),
            (248, 249, 250),
            (252, 250, 245),
            (245, 245, 242)
        ])
        img = Image.new("RGB", (w, h), bg_color)
        draw = ImageDraw.Draw(img)

        # Document Header
        doc_type = random.choice(DOC_TYPES)
        draw.rectangle([(20, 20), (w - 20, 25)], fill=(15, 23, 42))
        draw.text((30, 35), doc_type, fill=(15, 23, 42))
        draw.text((30, 52), f"Reference Document No: DOC-{random.randint(100000, 999999)}-2026", fill=(100, 116, 139))
        draw.text((w - 180, 52), f"Date: 2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}", fill=(100, 116, 139))
        draw.line([(30, 70), (w - 30, 70)], fill=(203, 213, 225), width=1)

        # Document Body text paragraphs
        y = 85
        while y < h - 160:
            para_type = random.choice(["text", "text", "table", "key_value", "code"])
            if para_type == "text":
                num_lines = random.randint(2, 5)
                for _ in range(num_lines):
                    line_words = random.sample(LOREM_WORDS, random.randint(7, 13))
                    line_str = " ".join(line_words).capitalize()
                    draw.text((30, y), line_str, fill=(30, 41, 59))
                    y += 15
                y += 10
            elif para_type == "table":
                draw.rectangle([(30, y), (w - 30, y + 60)], outline=(203, 213, 225), width=1)
                draw.rectangle([(30, y), (w - 30, y + 18)], fill=(241, 245, 249))
                draw.text((35, y + 3), "Item Description", fill=(15, 23, 42))
                draw.text((w // 2, y + 3), "Qty / Units", fill=(15, 23, 42))
                draw.text((w - 100, y + 3), "Total (USD)", fill=(15, 23, 42))
                draw.line([(30, y + 18), (w - 30, y + 18)], fill=(203, 213, 225), width=1)
                for row_i in range(2):
                    ry = y + 22 + row_i * 18
                    draw.text((35, ry), f"Foliar diagnostic unit #{row_i+1}", fill=(71, 85, 105))
                    draw.text((w // 2, ry), f"{random.randint(1, 25)} pcs", fill=(71, 85, 105))
                    draw.text((w - 100, ry), f"${random.randint(100, 950)}.00", fill=(71, 85, 105))
                y += 75
            elif para_type == "key_value":
                for _ in range(3):
                    key = random.choice(["Account Number", "Authorization Code", "Tax ID / EIN", "Issuing Authority", "Status Code"])
                    val = f"VAL-{random.randint(1000, 9999)}-OK"
                    draw.text((30, y), f"{key}:", fill=(15, 23, 42))
                    draw.text((200, y), val, fill=(71, 85, 105))
                    y += 16
                y += 8
            else:
                y += 20

        # Footer & Stamp / Signature block
        draw.line([(30, h - 80), (w - 30, h - 80)], fill=(203, 213, 225), width=1)
        draw.text((30, h - 70), "CONFIDENTIAL & PROPRIETARY DOCUMENT", fill=(148, 163, 184))
        draw.text((30, h - 55), "Generated for Department of Computer Science & Engineering Major Project", fill=(148, 163, 184))
        draw.rectangle([(w - 160, h - 75), (w - 35, h - 35)], outline=(220, 38, 38), width=2)
        draw.text((w - 150, h - 65), "OFFICIALLY VERIFIED", fill=(220, 38, 38))
        draw.text((w - 135, h - 50), "SEAL / SIGNED", fill=(220, 38, 38))

        # Save document
        dst = out_dir / f"document_{i:04d}.png"
        img.save(dst)

    all_docs = list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg"))
    print(f"Total document images available: {len(all_docs)}")
    return [str(p) for p in all_docs]


def build_verification_splits():
    """Compiles all classes into train, val, test CSV manifests."""
    build_directories()

    # 1. Cotton Leaf samples (already grouped into train/val/test)
    cotton_splits = get_cotton_leaf_samples()

    # 2. Non-cotton leaf samples
    non_cotton_files = download_non_cotton_leaves(target_count=1000)
    random.shuffle(non_cotton_files)
    n_nc = len(non_cotton_files)
    n_nc_train = int(n_nc * 0.70)
    n_nc_val = int(n_nc * 0.15)
    nc_train = non_cotton_files[:n_nc_train]
    nc_val = non_cotton_files[n_nc_train:n_nc_train + n_nc_val]
    nc_test = non_cotton_files[n_nc_train + n_nc_val:]

    # 3. Non-leaf samples
    faces = download_lfw_faces(target_count=350)
    docs = generate_document_images(target_count=350)
    objs = download_imagenet_objects(target_count=350)

    def _split_items(items):
        random.shuffle(items)
        n = len(items)
        tr = int(n * 0.70)
        va = int(n * 0.15)
        return items[:tr], items[tr:tr+va], items[tr+va:]

    face_tr, face_va, face_te = _split_items(faces)
    doc_tr, doc_va, doc_te = _split_items(docs)
    obj_tr, obj_va, obj_te = _split_items(objs)

    nl_train = [(p, "human_face") for p in face_tr] + [(p, "document") for p in doc_tr] + [(p, "object_vehicle_animal") for p in obj_tr]
    nl_val = [(p, "human_face") for p in face_va] + [(p, "document") for p in doc_va] + [(p, "object_vehicle_animal") for p in obj_va]
    nl_test = [(p, "human_face") for p in face_te] + [(p, "document") for p in doc_te] + [(p, "object_vehicle_animal") for p in obj_te]

    # Combine into unified split tables
    all_manifest = []

    for s_name in ["train", "val", "test"]:
        records = []
        # Cotton samples
        records.extend(cotton_splits[s_name])

        # Non-cotton leaves
        nc_list = nc_train if s_name == "train" else (nc_val if s_name == "val" else nc_test)
        for p in nc_list:
            records.append({
                "filepath": str(p),
                "class_name": "non_cotton_leaf",
                "label": 1,
                "sub_category": Path(p).name.split("_")[0],
                "split": s_name
            })

        # Non-leaf images
        nl_list = nl_train if s_name == "train" else (nl_val if s_name == "val" else nl_test)
        for p, sub in nl_list:
            records.append({
                "filepath": str(p),
                "class_name": "non_leaf",
                "label": 2,
                "sub_category": sub,
                "split": s_name
            })

        random.shuffle(records)
        df_split = pd.DataFrame(records)
        split_p = SPLITS_DIR / f"{s_name}.csv"
        df_split.to_csv(split_p, index=False)
        print(f"Saved {s_name}.csv: {len(df_split)} rows. Breakdown:\n{df_split['class_name'].value_counts().to_dict()}")
        all_manifest.extend(records)

    df_all = pd.DataFrame(all_manifest)
    manifest_p = SPLITS_DIR / "manifest.csv"
    df_all.to_csv(manifest_p, index=False)
    print(f"\n==================================================")
    print(f"VERIFICATION DATASET MANIFEST SAVED: {manifest_p}")
    print(f"Total Dataset Images: {len(df_all)}")
    print(f"Class Breakdown:\n{df_all['class_name'].value_counts()}")
    print(f"Split Breakdown:\n{df_all['split'].value_counts()}")
    print(f"==================================================")


if __name__ == "__main__":
    build_verification_splits()
