
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.paths import resolve_platform_path


# Keep this class order identical for training, validation and testing.
CLASS_NAMES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = 224

# ImageNet normalization values used with pretrained EfficientNet models.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(training=False, image_size=IMAGE_SIZE):
    """
    Create image transformations.

    Training:
        Resize, random crop, horizontal flip, slight rotation,
        mild colour augmentation and normalization.

    Validation/testing:
        Resize and normalization only.
    """

    if training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.08,
                hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CottonLeafDataset(Dataset):
    """
    PyTorch Dataset for the cotton leaf classification project.

    Expected CSV columns:
        - absolute_path
        - class_folder_label

    The generated manifest also contains a 'relative_path' column,
    which can be used as a fallback when absolute_path is unavailable.
    """

    def __init__(
        self,
        csv_path,
        transform=None,
        class_to_idx=None,
        dataset_root=None,
        return_path=False,
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset CSV file not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        required_columns = {"class_folder_label"}

        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                "CSV must contain the 'class_folder_label' column."
            )

        if "absolute_path" not in self.data.columns:
            if "relative_path" not in self.data.columns:
                raise ValueError(
                    "CSV must contain either 'absolute_path' "
                    "or 'relative_path'."
                )

        self.class_to_idx = (
            class_to_idx.copy()
            if class_to_idx is not None
            else CLASS_TO_IDX.copy()
        )

        self.idx_to_class = {
            index: name
            for name, index in self.class_to_idx.items()
        }

        self.dataset_root = (
            Path(dataset_root) if dataset_root is not None else None
        )

        self.transform = transform
        self.return_path = return_path

        # Validate class labels before training.
        unknown_classes = (
            set(self.data["class_folder_label"].dropna().unique())
            - set(self.class_to_idx.keys())
        )

        if unknown_classes:
            raise ValueError(
                f"Unknown classes found in CSV: {unknown_classes}"
            )

        if self.data["class_folder_label"].isna().any():
            raise ValueError("CSV contains missing class labels.")

        # Resolve image paths across Windows and WSL/POSIX filesystems.
        if "absolute_path" in self.data.columns:
            self.image_paths = [
                str(resolve_platform_path(path))
                for path in self.data["absolute_path"].astype(str)
            ]
        else:
            root_path = resolve_platform_path(self.dataset_root)
            self.image_paths = [
                str(root_path / path)
                for path in self.data["relative_path"].astype(str)
            ]

        self.labels = [
            self.class_to_idx[label]
            for label in self.data["class_folder_label"]
        ]

        # Check that all image paths exist.
        missing_images = [
            path for path in self.image_paths
            if not Path(path).is_file()
        ]

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} image files were not found. "
                f"Examples: {missing_images[:5]}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return image, label, image_path

        return image, label


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    manifest_dir = (
        project_root / "artifacts" / "manifests" / "splits"
    )

    train_csv = manifest_dir / "train.csv"
    validation_csv = manifest_dir / "validation.csv"
    test_csv = manifest_dir / "test.csv"

    train_dataset = CottonLeafDataset(
        csv_path=train_csv,
        transform=build_transforms(training=True),
    )

    validation_dataset = CottonLeafDataset(
        csv_path=validation_csv,
        transform=build_transforms(training=False),
    )

    test_dataset = CottonLeafDataset(
        csv_path=test_csv,
        transform=build_transforms(training=False),
    )

    print("\nCOTTON LEAF DATASET VERIFICATION")
    print("-" * 40)

    print("Classes:", len(CLASS_NAMES))
    print("Class mapping:", CLASS_TO_IDX)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))
    print("Testing images:", len(test_dataset))

    image, label = train_dataset[0]

    print("\nSample verification:")
    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)
    print("Label index:", label)
    print("Class name:", train_dataset.idx_to_class[label])
    print("Pixel value range:",
          (image.min().item(), image.max().item()))

    print("\nDataset loading: SUCCESS")
