

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageTk, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Pillow is required. Activate the project environment and run: "
        "python -m pip install Pillow"
    ) from exc

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Tkinter is not available in this Python installation. "
        "On Windows, reinstall Python with the Tcl/Tk support option enabled."
    ) from exc


@dataclass
class ReviewPair:
    pair_id: str
    pair_type: str
    image_a: str
    label_a: str
    image_b: str
    label_b: str
    similarity_info: str = ""


CSV_FIELDS = [
    "pair_id",
    "pair_type",
    "image_a",
    "label_a",
    "image_b",
    "label_b",
    "similarity_info",
    "decision",
    "reviewer_notes",
    "reviewed_at_utc",
]

DECISION_OPTIONS = [
    "Unreviewed",
    "Same image; labels agree",
    "Same image; labels conflict",
    "Same source/near-identical image; labels agree",
    "Different images; labels appear correct",
    "Possible label issue; needs expert review",
    "Unsure; needs further review",
]


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Required audit file not found: {csv_path}\n"
            "Run scripts/audit_dataset.py first."
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_relative_path(value: str) -> str:
    """Normalize CSV paths produced on Windows or POSIX systems."""
    return value.strip().replace("\\", "/").lstrip("/")


def build_review_pairs(audit_dir: Path) -> list[ReviewPair]:
    pairs: list[ReviewPair] = []

    # Exact duplicate CSV has one row per file, grouped by duplicate_group.
    exact_rows = read_csv_rows(audit_dir / "exact_duplicates.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in exact_rows:
        group_id = row.get("duplicate_group", "").strip()
        if group_id:
            groups.setdefault(group_id, []).append(row)

    for group_id, rows in sorted(groups.items(), key=lambda item: item[0]):
        # Pair all files in a group. For the current dataset, groups contain two
        # files, but combinations also handles larger duplicate groups safely.
        for pair_index, (left, right) in enumerate(itertools.combinations(rows, 2), 1):
            path_a = normalize_relative_path(left.get("relative_path", ""))
            path_b = normalize_relative_path(right.get("relative_path", ""))
            label_a = path_a.split("/")[0] if "/" in path_a else ""
            label_b = path_b.split("/")[0] if "/" in path_b else ""
            pairs.append(
                ReviewPair(
                    pair_id=f"EXACT-{group_id}-{pair_index}",
                    pair_type="Exact duplicate",
                    image_a=path_a,
                    label_a=label_a,
                    image_b=path_b,
                    label_b=label_b,
                    similarity_info=f"Same SHA-256 group: {group_id}",
                )
            )

    # Near duplicate CSV already contains pair rows and folder labels.
    near_rows = read_csv_rows(audit_dir / "near_duplicates.csv")
    for index, row in enumerate(near_rows, 1):
        distance = row.get("phash_hamming_distance", "")
        pairs.append(
            ReviewPair(
                pair_id=f"NEAR-{index:04d}",
                pair_type="Near duplicate",
                image_a=normalize_relative_path(row.get("image_a", "")),
                label_a=row.get("label_a", "").strip(),
                image_b=normalize_relative_path(row.get("image_b", "")),
                label_b=row.get("label_b", "").strip(),
                similarity_info=f"Perceptual-hash Hamming distance: {distance}",
            )
        )

    return pairs


def safe_image_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve an image path and prevent paths escaping the dataset directory."""
    root = dataset_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Unsafe image path outside dataset root: {relative_path}") from exc
    return candidate


class DuplicateReviewApp:
    def __init__(self, root: tk.Tk, dataset_root: Path, audit_dir: Path,
                 pairs: list[ReviewPair], decisions_path: Path):
        self.root = root
        self.dataset_root = dataset_root.resolve()
        self.audit_dir = audit_dir.resolve()
        self.pairs = pairs
        self.decisions_path = decisions_path
        self.index = 0
        self.photo_a: Optional[ImageTk.PhotoImage] = None
        self.photo_b: Optional[ImageTk.PhotoImage] = None
        self.saved: dict[str, dict[str, str]] = self.load_existing_decisions()

        self.root.title("Cotton Leaf AI - Duplicate Review")
        self.root.geometry("1240x850")
        self.root.minsize(900, 650)

        self.title_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        self.path_a_var = tk.StringVar()
        self.path_b_var = tk.StringVar()
        self.decision_var = tk.StringVar(value="Unreviewed")
        self.status_var = tk.StringVar(value="Ready. Original images are read-only.")

        self._build_ui()
        self.show_pair(0)

    def load_existing_decisions(self) -> dict[str, dict[str, str]]:
        existing: dict[str, dict[str, str]] = {}
        if self.decisions_path.is_file():
            try:
                with self.decisions_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        pair_id = row.get("pair_id", "")
                        if pair_id:
                            existing[pair_id] = row
            except (OSError, csv.Error) as exc:
                messagebox.showwarning(
                    "Could not load previous decisions",
                    f"A new review session will start. Details: {exc}",
                )
        return existing

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, textvariable=self.title_var,
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.meta_var,
                  font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 10))

        image_frame = ttk.Frame(outer)
        image_frame.pack(fill="both", expand=True)
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)
        image_frame.rowconfigure(1, weight=1)

        ttk.Label(image_frame, text="IMAGE A", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, pady=4
        )
        ttk.Label(image_frame, text="IMAGE B", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=1, pady=4
        )

        self.image_a_label = ttk.Label(image_frame, anchor="center", relief="groove")
        self.image_b_label = ttk.Label(image_frame, anchor="center", relief="groove")
        self.image_a_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=4)
        self.image_b_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=4)

        details = ttk.Frame(outer)
        details.pack(fill="x", pady=(8, 4))
        ttk.Label(details, textvariable=self.path_a_var, wraplength=570).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )
        ttk.Label(details, textvariable=self.path_b_var, wraplength=570).grid(
            row=0, column=1, sticky="w"
        )

        decision_frame = ttk.LabelFrame(outer, text="Review decision", padding=10)
        decision_frame.pack(fill="x", pady=8)
        ttk.Label(decision_frame, text="Decision:").grid(row=0, column=0, sticky="w")
        self.decision_combo = ttk.Combobox(
            decision_frame,
            textvariable=self.decision_var,
            values=DECISION_OPTIONS,
            state="readonly",
            width=52,
        )
        self.decision_combo.grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(decision_frame, text="Reviewer notes:").grid(
            row=1, column=0, sticky="nw", pady=(8, 0)
        )
        self.notes_text = tk.Text(decision_frame, height=3, wrap="word")
        self.notes_text.grid(row=1, column=1, sticky="ew", padx=8, pady=(8, 0))
        decision_frame.columnconfigure(1, weight=1)

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=6)
        ttk.Button(buttons, text="◀ Previous", command=self.previous_pair).pack(side="left")
        ttk.Button(buttons, text="Save decision", command=self.save_decision).pack(
            side="left", padx=8
        )
        ttk.Button(buttons, text="Next ▶", command=self.next_pair).pack(side="left")
        ttk.Button(buttons, text="Save & Next", command=self.save_and_next).pack(
            side="left", padx=8
        )
        ttk.Button(buttons, text="Open output folder", command=self.open_output_folder).pack(
            side="right"
        )

        ttk.Label(outer, textvariable=self.status_var, foreground="#176b3a").pack(
            anchor="w", pady=(6, 0)
        )
        ttk.Label(
            outer,
            text="Safety: this tool only reads source images. Decisions are saved to the audit folder, never to the dataset.",
            wraplength=1150,
        ).pack(anchor="w", pady=(4, 0))

    def _load_thumbnail(self, relative_path: str) -> tuple[Optional[ImageTk.PhotoImage], str]:
        try:
            path = safe_image_path(self.dataset_root, relative_path)
            if not path.is_file():
                return None, f"FILE NOT FOUND\n{path}"
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((540, 460), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image.copy())
            return photo, ""
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            return None, f"Could not display image:\n{exc}"

    def show_pair(self, index: int) -> None:
        if not self.pairs:
            self.title_var.set("No duplicate pairs to review")
            self.meta_var.set("The exact_duplicates.csv and near_duplicates.csv files contain no pairs.")
            return

        self.index = max(0, min(index, len(self.pairs) - 1))
        pair = self.pairs[self.index]
        self.title_var.set(f"{pair.pair_type} — {self.index + 1} of {len(self.pairs)}")
        self.meta_var.set(
            f"{pair.pair_id}    |    {pair.similarity_info}    |    "
            f"Folder labels: {pair.label_a}  ↔  {pair.label_b}"
        )
        self.path_a_var.set(f"A — {pair.label_a}\n{pair.image_a}")
        self.path_b_var.set(f"B — {pair.label_b}\n{pair.image_b}")

        self.photo_a, error_a = self._load_thumbnail(pair.image_a)
        self.photo_b, error_b = self._load_thumbnail(pair.image_b)
        self.image_a_label.configure(
            image=self.photo_a if self.photo_a else "",
            text=error_a,
        )
        self.image_b_label.configure(
            image=self.photo_b if self.photo_b else "",
            text=error_b,
        )

        previous = self.saved.get(pair.pair_id, {})
        self.decision_var.set(previous.get("decision") or "Unreviewed")
        self.notes_text.delete("1.0", "end")
        self.notes_text.insert("1.0", previous.get("reviewer_notes", ""))
        self.status_var.set(
            f"Saved decisions: {sum(1 for row in self.saved.values() if row.get('decision') not in ('', 'Unreviewed'))}"
            f" / {len(self.pairs)}. Source images are read-only."
        )

    def _current_record(self) -> dict[str, str]:
        pair = self.pairs[self.index]
        decision = self.decision_var.get().strip() or "Unreviewed"
        notes = self.notes_text.get("1.0", "end").strip()
        old = self.saved.get(pair.pair_id, {})
        reviewed_at = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            if decision != "Unreviewed" or notes
            else old.get("reviewed_at_utc", "")
        )
        return {
            "pair_id": pair.pair_id,
            "pair_type": pair.pair_type,
            "image_a": pair.image_a,
            "label_a": pair.label_a,
            "image_b": pair.image_b,
            "label_b": pair.label_b,
            "similarity_info": pair.similarity_info,
            "decision": decision,
            "reviewer_notes": notes,
            "reviewed_at_utc": reviewed_at,
        }

    def _write_decisions(self) -> None:
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.decisions_path.with_suffix(".tmp")
        # Keep a row for every detected pair, including those not yet reviewed.
        rows: list[dict[str, str]] = []
        current_by_id = {pair.pair_id: pair for pair in self.pairs}
        for pair in self.pairs:
            saved = self.saved.get(pair.pair_id)
            if saved:
                rows.append({field: saved.get(field, "") for field in CSV_FIELDS})
            else:
                rows.append({
                    "pair_id": pair.pair_id,
                    "pair_type": pair.pair_type,
                    "image_a": pair.image_a,
                    "label_a": pair.label_a,
                    "image_b": pair.image_b,
                    "label_b": pair.label_b,
                    "similarity_info": pair.similarity_info,
                    "decision": "Unreviewed",
                    "reviewer_notes": "",
                    "reviewed_at_utc": "",
                })
        # Atomic replacement affects only the output CSV, never original images.
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(self.decisions_path)

    def save_decision(self) -> None:
        if not self.pairs:
            return
        record = self._current_record()
        self.saved[record["pair_id"]] = record
        try:
            self._write_decisions()
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save review decisions:\n{exc}")
            return
        self.status_var.set(f"Saved {record['pair_id']} to {self.decisions_path}")

    def save_and_next(self) -> None:
        self.save_decision()
        if self.index < len(self.pairs) - 1:
            self.show_pair(self.index + 1)

    def previous_pair(self) -> None:
        if self.index > 0:
            self.show_pair(self.index - 1)

    def next_pair(self) -> None:
        if self.index < len(self.pairs) - 1:
            self.show_pair(self.index + 1)

    def open_output_folder(self) -> None:
        # Avoid shell execution; use the standard Windows file explorer only.
        if sys.platform == "win32":  # pragma: no cover - Windows-only
            import os
            os.startfile(str(self.audit_dir))  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Output folder", str(self.audit_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review exact and near-duplicate cotton leaf images side by side."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Main_Project\dataset\Original Dataset"),
        help="Read-only root folder containing the seven class folders.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("artifacts/audit"),
        help="Folder containing exact_duplicates.csv and near_duplicates.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to audit-dir/duplicate_review_decisions.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    output_path = (args.output.expanduser().resolve() if args.output else
                   audit_dir / "duplicate_review_decisions.csv")

    if not dataset_root.is_dir():
        print(f"ERROR: Dataset directory does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not audit_dir.is_dir():
        print(f"ERROR: Audit directory does not exist: {audit_dir}", file=sys.stderr)
        return 2

    try:
        pairs = build_review_pairs(audit_dir)
    except (OSError, csv.Error, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not pairs:
        print("No exact or near-duplicate pairs were found in the audit CSV files.")
        return 0

    print(f"Loaded {len(pairs)} review pairs.")
    print(f"Dataset (read-only): {dataset_root}")
    print(f"Review decisions will be saved to: {output_path}")
    print("No source image will be changed.")

    root = tk.Tk()
    DuplicateReviewApp(root, dataset_root, audit_dir, pairs, output_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
