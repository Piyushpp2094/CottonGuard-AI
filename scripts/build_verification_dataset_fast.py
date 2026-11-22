"""
High-Speed Verification Dataset Builder:
1. Cotton Leaves: References the 2,132 SAR-CLD-2024 images directly with exact train/val/test splits.
2. Non-Cotton Leaves:
   - 128 Bean leaves (healthy, angular leaf spot, rust) from tmp_beans
   - 150 Potato leaves from PlantVillage URLs
   - 250 Tomato healthy & early blight leaves from PlantVillage URLs
   - 250 Apple leaves from PlantVillage URLs
   Total: ~780 non-cotton leaves.
3. Non-Leaf Images:
   - 350 Human faces from LFW
   - 350 Generated authentic document/paper/invoice images
   - 350 ImageNet objects/animals/vehicles from EliSchwartz/imagenet-sample-images
   Total: 1,050 non-leaf images.
"""

import io
import json
import os
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VERIF_DIR = DATA_DIR / "verification"
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
    print("Created verification directories.", flush=True)


def download_file(url, dst):
    if dst.is_file() and dst.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        if len(data) > 1000:
            with open(dst, "wb") as f:
                f.write(data)
            return True
        return False
    except Exception:
        return False


def get_cotton_samples():
    """Reads existing group-isolated splits from artifacts/manifests/splits/."""
    splits = {}
    split_files = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
    for s_name, fname in split_files.items():
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
                    abs_p = alt_p if alt_p.is_file() else full_p
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
        print(f"Cotton leaf {s_name}: {len(samples)} images.", flush=True)
    return splits


def populate_non_cotton_leaves():
    out_dir = IMAGES_DIR / "non_cotton_leaf"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy beans leaves
    beans_tmp = DATA_DIR / "tmp_beans"
    if beans_tmp.is_dir():
        for f in beans_tmp.rglob("*.jpg"):
            dst = out_dir / f"bean_{f.name}"
            if not dst.is_file():
                shutil.copy2(f, dst)

    # 2. Download from plantvillage_urls.json
    pv_json = DATA_DIR / "plantvillage_urls.json"
    if pv_json.is_file():
        with open(pv_json) as f:
            pv_urls = json.load(f)

        tasks = []
        # Target: 150 potato, 200 tomato early, 200 tomato healthy, 200 apple healthy
        targets = {
            "Potato___healthy": 150,
            "Tomato___Early_blight": 200,
            "Tomato___healthy": 200,
            "Apple___healthy": 200,
        }
        for cat, count in targets.items():
            urls = pv_urls.get(cat, [])[:count]
            for i, u in enumerate(urls):
                dst = out_dir / f"{cat}_{i:04d}.jpg"
                tasks.append((u, dst))

        print(f"Downloading {len(tasks)} non-cotton leaves from PlantVillage...", flush=True)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(download_file, u, d) for u, d in tasks]
            for _ in as_completed(futures):
                pass

    all_nc = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    print(f"Total non-cotton leaves ready: {len(all_nc)}", flush=True)
    return [str(p) for p in all_nc]


def populate_faces(target_count=350):
    out_dir = IMAGES_DIR / "non_leaf" / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"Faces already present: {len(existing)}", flush=True)
        return [str(p) for p in existing]

    print("Extracting human faces from LFW parquet...", flush=True)
    url = "https://huggingface.co/datasets/logasja/lfw/resolve/main/data/train-00000-of-00001.parquet"
    df = pd.read_parquet(url, columns=["image"])
    for i in range(min(target_count, len(df))):
        dst = out_dir / f"face_{i:04d}.jpg"
        if not dst.is_file():
            with open(dst, "wb") as f:
                f.write(df["image"].iloc[i]["bytes"])

    all_faces = list(out_dir.glob("*.jpg"))
    print(f"Total human faces ready: {len(all_faces)}", flush=True)
    return [str(p) for p in all_faces]


def populate_objects(target_count=350):
    out_dir = IMAGES_DIR / "non_leaf" / "objects"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.JPEG")) + list(out_dir.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"Objects already present: {len(existing)}", flush=True)
        return [str(p) for p in existing]

    print("Downloading ImageNet objects/animals/vehicles from GitHub...", flush=True)
    api_url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read())

    image_items = [x for x in items if x["name"].lower().endswith((".jpeg", ".jpg", ".png"))]
    random.shuffle(image_items)
    to_download = image_items[:target_count]

    tasks = [(it["download_url"], out_dir / it["name"]) for it in to_download]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_file, u, d) for u, d in tasks]
        for _ in as_completed(futures):
            pass

    all_objs = list(out_dir.glob("*.JPEG")) + list(out_dir.glob("*.jpg"))
    print(f"Total objects ready: {len(all_objs)}", flush=True)
    return [str(p) for p in all_objs]


def generate_documents(target_count=350):
    out_dir = IMAGES_DIR / "non_leaf" / "documents"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.png"))
    if len(existing) >= target_count:
        print(f"Documents already present: {len(existing)}", flush=True)
        return [str(p) for p in existing]

    print(f"Generating {target_count - len(existing)} realistic document images...", flush=True)
    DOC_TYPES = [
        "TAX INVOICE & RECEIPT",
        "RESEARCH PAPER DRAFT",
        "LEGAL AGREEMENT & NDA",
        "MEDICAL LAB REPORT",
        "PURCHASE ORDER VOUCHER",
        "BANK ACCOUNT STATEMENT",
        "CERTIFICATE OF ANALYSIS",
        "OFFICIAL MEMORANDUM"
    ]
    WORDS = (
        "cotton pathology agricultural advisory diagnostic report analysis "
        "severity percentage calculation verification gate classification "
        "invoice total amount due tax terms signature authorized stamp "
        "department computer science engineering major project evaluation "
        "bacterial blight curl virus healthy leaf herbicide damage jassids "
        "leaf redding variegation model weights checkpoint pytorch cuda "
        "table column header summary description status approved rejected"
    ).split()

    start_idx = len(existing)
    for i in range(start_idx, target_count):
        w, h = random.choice([(600, 800), (700, 950), (650, 850)])
        bg = random.choice([(255, 255, 255), (250, 250, 248), (248, 249, 250)])
        img = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(img)

        # Header bar & title
        doc_type = random.choice(DOC_TYPES)
        draw.rectangle([(20, 20), (w - 20, 25)], fill=(30, 41, 59))
        draw.text((30, 35), doc_type, fill=(15, 23, 42))
        draw.text((30, 52), f"Doc Ref: DOC-2026-{random.randint(10000, 99999)}", fill=(100, 116, 139))
        draw.text((w - 180, 52), f"Date: 2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}", fill=(100, 116, 139))
        draw.line([(30, 70), (w - 30, 70)], fill=(203, 213, 225), width=1)

        # Content blocks
        y = 85
        while y < h - 140:
            block = random.choice(["para", "para", "table", "kv"])
            if block == "para":
                for _ in range(random.randint(2, 4)):
                    line = " ".join(random.sample(WORDS, random.randint(7, 12))).capitalize()
                    draw.text((30, y), line, fill=(51, 65, 85))
                    y += 15
                y += 10
            elif block == "table":
                draw.rectangle([(30, y), (w - 30, y + 55)], outline=(203, 213, 225), width=1)
                draw.rectangle([(30, y), (w - 30, y + 16)], fill=(241, 245, 249))
                draw.text((35, y + 2), "Description", fill=(15, 23, 42))
                draw.text((w // 2, y + 2), "Reference", fill=(15, 23, 42))
                draw.text((w - 100, y + 2), "Amount", fill=(15, 23, 42))
                draw.line([(30, y + 16), (w - 30, y + 16)], fill=(203, 213, 225), width=1)
                for r_idx in range(2):
                    ry = y + 20 + r_idx * 16
                    draw.text((35, ry), f"Foliar diagnostic test #{r_idx+1}", fill=(71, 85, 105))
                    draw.text((w // 2, ry), f"BATCH-{random.randint(100, 999)}", fill=(71, 85, 105))
                    draw.text((w - 100, ry), f"${random.randint(100, 900)}.00", fill=(71, 85, 105))
                y += 70
            elif block == "kv":
                for _ in range(3):
                    k = random.choice(["Authorization", "Tax ID", "Operator", "Status", "Account"])
                    v = f"REF-{random.randint(1000, 9999)}-OK"
                    draw.text((30, y), f"{k}:", fill=(15, 23, 42))
                    draw.text((180, y), v, fill=(71, 85, 105))
                    y += 16
                y += 8
            else:
                y += 20

        # Footer & Stamp
        draw.line([(30, h - 70), (w - 30, h - 70)], fill=(203, 213, 225), width=1)
        draw.text((30, h - 60), "CONFIDENTIAL DOCUMENT • FOR OFFICIAL AUDIT ONLY", fill=(148, 163, 184))
        draw.rectangle([(w - 150, h - 65), (w - 30, h - 25)], outline=(220, 38, 38), width=2)
        draw.text((w - 140, h - 55), "VERIFIED & SIGNED", fill=(220, 38, 38))
        draw.text((w - 130, h - 40), "AUTHORIZED", fill=(220, 38, 38))

        dst = out_dir / f"document_{i:04d}.png"
        img.save(dst)

    all_docs = list(out_dir.glob("*.png"))
    print(f"Total documents ready: {len(all_docs)}", flush=True)
    return [str(p) for p in all_docs]


def build_verification_dataset():
    t0 = time.time()
    build_directories()

    # 1. Cotton leaves
    cotton_splits = get_cotton_samples()

    # 2. Non-cotton leaves
    nc_files = populate_non_cotton_leaves()
    random.shuffle(nc_files)
    n_nc = len(nc_files)
    n_tr = int(n_nc * 0.70)
    n_va = int(n_nc * 0.15)
    nc_splits = {
        "train": nc_files[:n_tr],
        "val": nc_files[n_tr:n_tr + n_va],
        "test": nc_files[n_tr + n_va:],
    }

    # 3. Non-leaf
    face_files = populate_faces(target_count=350)
    doc_files = generate_documents(target_count=350)
    obj_files = populate_objects(target_count=350)

    def _split_3(files):
        random.shuffle(files)
        n = len(files)
        tr = int(n * 0.70)
        va = int(n * 0.15)
        return files[:tr], files[tr:tr+va], files[tr+va:]

    f_tr, f_va, f_te = _split_3(face_files)
    d_tr, d_va, d_te = _split_3(doc_files)
    o_tr, o_va, o_te = _split_3(obj_files)

    nl_splits = {
        "train": [(p, "human_face") for p in f_tr] + [(p, "document") for p in d_tr] + [(p, "object_vehicle_animal") for p in o_tr],
        "val": [(p, "human_face") for p in f_va] + [(p, "document") for p in d_va] + [(p, "object_vehicle_animal") for p in o_va],
        "test": [(p, "human_face") for p in f_te] + [(p, "document") for p in d_te] + [(p, "object_vehicle_animal") for p in o_te],
    }

    all_records = []
    for s_name in ["train", "val", "test"]:
        records = []
        # Cotton
        records.extend(cotton_splits[s_name])
        # Non-cotton leaves
        for p in nc_splits[s_name]:
            records.append({
                "filepath": str(p),
                "class_name": "non_cotton_leaf",
                "label": 1,
                "sub_category": Path(p).name.split("_")[0],
                "split": s_name,
            })
        # Non-leaf
        for p, sub in nl_splits[s_name]:
            records.append({
                "filepath": str(p),
                "class_name": "non_leaf",
                "label": 2,
                "sub_category": sub,
                "split": s_name,
            })

        random.shuffle(records)
        df_split = pd.DataFrame(records)
        split_p = SPLITS_DIR / f"{s_name}.csv"
        df_split.to_csv(split_p, index=False)
        print(f"Saved {s_name}.csv: {len(df_split)} rows. Counts: {df_split['class_name'].value_counts().to_dict()}", flush=True)
        all_records.extend(records)

    df_manifest = pd.DataFrame(all_records)
    manifest_p = SPLITS_DIR / "manifest.csv"
    df_manifest.to_csv(manifest_p, index=False)

    print("\n=======================================================", flush=True)
    print("COTTON LEAF VERIFICATION DATASET SUCCESSFULLY BUILT!", flush=True)
    print(f"Total Images: {len(df_manifest)} in {round(time.time() - t0, 1)}s", flush=True)
    print(f"Class Distribution:\n{df_manifest['class_name'].value_counts()}", flush=True)
    print(f"Split Distribution:\n{df_manifest['split'].value_counts()}", flush=True)
    print(f"Manifest Path: {manifest_p}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    build_verification_dataset()
