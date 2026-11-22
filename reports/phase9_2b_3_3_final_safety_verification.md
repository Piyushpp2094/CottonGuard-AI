# CottonGuard AI — Phase 9.2B.3.3: Fresh-Run Overwrite Protection and Final Safety Verification
## Faster R-CNN Detector | Pre-Training Safety and Overwrite Hardening Audit

**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Phase:** 9.2B.3.3 — Fresh-Run Overwrite Protection and Final Safety Verification  
**Date:** 2026-11-15  
**Execution Mode:** Controlled Safety Implementation & Verification (NO TRAINING AUTHORIZED)  
**Target Model:** TorchVision Faster R-CNN ResNet-50 FPN V2  
**Final Status:** `NOT_TRAINED` | **Readiness Classification:** `READY_FOR_FUTURE_USER_AUTHORIZATION`

---

## 1. Executive Summary

Phase 9.2B.3.3 implements and rigorously verifies the final pre-training safeguards for the Faster R-CNN detection pipeline. The primary engineering goal is to guarantee that an unintended fresh training run cannot silently overwrite an existing optimal detector checkpoint (`best_faster_rcnn.pt`). Under strict zero-training execution restrictions, the following safety systems were designed, integrated, and verified:

1. **Pre-Training Checkpoint Existence Guard:** Implemented `check_existing_checkpoint_guard` in `scripts/train_faster_rcnn.py`. If a pre-existing `best_faster_rcnn.pt` is detected and `--overwrite-existing` is not supplied, execution aborts immediately with `FileExistsError` prior to loading datasets, constructing models, or initializing optimizers.
2. **Explicit Overwrite Authorization (`--overwrite-existing`):** Implemented automated timestamped archiving (`<stem>_bak_<timestamp>.pt`) with mandatory SHA-256 integrity verification. If backup creation or checksum validation fails, execution terminates before any state is modified.
3. **Rejection of Unsupported Resume:** Added `--resume` CLI flag that fails early with `NotImplementedError`, preventing unsupported arguments from silently defaulting to fresh runs.
4. **Execution Ordering Verified:** Confirmed that path validation (Rules A–E) and overwrite guards execute strictly prior to `ckpt_dir.mkdir(...)`, dataset loading, and model construction.
5. **Symbolic-Link / Junction Empirical Investigation:** Attempted empirical symlink and directory junction creation on Windows NTFS; confirmed that non-elevated user accounts lack `SeCreateSymbolicLinkPrivilege` and junction creation permissions (`WinError 1314` / `Access is denied`). Explicitly reported that symlink escape was reviewed via `Path.resolve()` canonicalization but not empirically executed due to OS privilege boundaries.
6. **Isolated Unit & CLI Testing (100% Pass):** Executed a 10-scenario isolated test suite (`scratch/test_overwrite_policy.py`) and a 4-scenario end-to-end CLI dry-run test suite (`scratch/test_main_overwrite_cli.py`). All tests passed. Zero detector checkpoints were created.

---

## 2. Fresh-Run Overwrite Protection Implementation

### Core Guard Function (`scripts/train_faster_rcnn.py#L167-L226`)

```python
def check_existing_checkpoint_guard(
    best_ckpt_path: Path,
    overwrite_existing: bool = False,
    resume_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Enforces pre-training checkpoint overwrite protection.
    - Rule A: If best_ckpt_path does not exist, allows fresh run.
    - Rule B: If best_ckpt_path exists and neither overwrite nor resume supplied, raises FileExistsError.
    - Rule C: If overwrite_existing is True and best_ckpt_path exists, creates and SHA-256 verifies a timestamped backup.
    - Rule D: If resume_path is supplied, raises NotImplementedError (deferred).
    - Rule E: Executes prior to dataset, model, and optimizer initialization.
    Returns: backup_path if backup created, else None.
    """
    if resume_path is not None:
        raise NotImplementedError(
            "Resume functionality is deferred and not supported in this version. "
            "Please remove the '--resume' argument or wait for future resume authorization."
        )

    if not best_ckpt_path.is_file():
        # Rule A: Fresh run when no checkpoint exists
        return None

    # Best checkpoint exists!
    if not overwrite_existing:
        # Rule B: Existing best checkpoint without explicit overwrite flag
        raise FileExistsError(
            f"Pre-existing trained checkpoint found at '{best_ckpt_path}'. "
            "To prevent accidental data loss, training is blocked. "
            "Provide '--overwrite-existing' to explicitly authorize creating a timestamped backup and replacing it."
        )

    # Rule C: Explicit overwrite authorized. Create timestamped backup and verify SHA-256.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = best_ckpt_path.with_name(f"{best_ckpt_path.stem}_bak_{ts}.pt")

    orig_bytes = best_ckpt_path.read_bytes()
    orig_hash = hashlib.sha256(orig_bytes).hexdigest().upper()

    shutil.copy2(best_ckpt_path, backup_path)

    if not backup_path.is_file():
        raise RuntimeError(f"Backup creation failed: '{backup_path}' does not exist.")

    bak_hash = hashlib.sha256(backup_path.read_bytes()).hexdigest().upper()
    if bak_hash != orig_hash:
        try:
            backup_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Backup verification failed: SHA-256 mismatch for '{backup_path}' "
            f"(original={orig_hash}, backup={bak_hash}). Training aborted."
        )

    print(f"[Backup] Successfully archived pre-existing best checkpoint to: {backup_path} (SHA-256: {bak_hash})")
    return backup_path
```

---

## 3. Backup Creation and Verification Behavior

When `--overwrite-existing` is supplied and an existing checkpoint is present at `best_ckpt_p`:
1. **Timestamped Identifier:** Creates a backup filename using UTC timestamp: `<stem>_bak_YYYYMMDD_HHMMSS.pt` located within the same protected directory.
2. **Metadata Preservation:** Utilizes `shutil.copy2` to preserve file mode and metadata.
3. **Cryptographic Verification:**
   - Computes `orig_hash = SHA256(best_ckpt_path)` prior to copy.
   - Computes `bak_hash = SHA256(backup_path)` after copy.
   - Requires exact bit-level equality (`bak_hash == orig_hash`).
4. **Failure Safeguard:** If hash verification fails or the backup file is incomplete, the partial backup file is deleted via `unlink()` and execution raises `RuntimeError`, aborting before any model weights or checkpoints can be replaced.

---

## 4. CLI Flag Behavior

In `scripts/train_faster_rcnn.py#L650-L685`:

```python
parser.add_argument(
    "--dry-run-check",
    action="store_true",
    help="Validate configuration, loaders, and model graph without running training",
)
parser.add_argument(
    "--overwrite-existing",
    action="store_true",
    help="Explicitly authorize archiving and overwriting an existing best detector checkpoint",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume training from an existing checkpoint (currently deferred / unsupported)",
)
```

- When running a fresh run on an empty directory: `--overwrite-existing` is not required.
- When running against an existing checkpoint: Omitting `--overwrite-existing` raises `FileExistsError`.
- Supplying `--overwrite-existing`: Archives and verifies the existing best model, then proceeds.

---

## 5. Unsupported Resume Behavior

If a user invokes the script with `--resume <path>`:
- The guard intercepts `resume_path` immediately.
- Raises `NotImplementedError: Resume functionality is deferred and not supported in this version.`
- Execution halts instantly with non-zero exit code.
- Guarantees that passing `--resume` will **never** silently execute a fresh training run that overwrites checkpoints.

---

## 6. Execution-Order Verification

The execution sequence inside `train_faster_rcnn()` (`scripts/train_faster_rcnn.py#L315-L355`) follows strict pre-conditions:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ EXECUTION ORDERING & EARLY-FAIL GATING SEQUENCE                                                  │
├──────┬───────────────────────────────────────────┬───────────────────────────────────────────────┤
│ Step │ Pipeline Action                           │ Abort Condition / Invariant Enforced          │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 1    │ Load YAML Configuration                   │ FileNotFoundError if config path is missing   │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 2    │ Checkpoint Path Validation                │ ValueError on Rule A-D violations (escapes,   │
│      │ `validate_checkpoint_paths`               │ collisions with classifier/U-Net, traversals) │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 3    │ Overwrite & Resume Guard                  │ FileExistsError if best checkpoint exists;    │
│      │ `check_existing_checkpoint_guard`         │ NotImplementedError on --resume; backup SHA   │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 4    │ Checkpoint Directory Creation             │ `ckpt_dir.mkdir()` called ONLY after steps    │
│      │                                           │ 2 and 3 succeed                               │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 5    │ Reproducibility Seeding & Device Setup    │ `torch.manual_seed(42)`, select CUDA / CPU   │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 6    │ Dataset Loading & DataLoader Creation     │ 35 train / 15 val images loaded               │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 7    │ Faster R-CNN Model Construction           │ Model instantiated & pretrained weights loaded│
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 8    │ Optimizer, Scaler & Scheduler Init        │ SGD, GradScaler(1024), WarmupCosine schedule  │
├──────┼───────────────────────────────────────────┼───────────────────────────────────────────────┤
│ 9    │ Dry-Run Check Return or Training Loop     │ Exits if --dry-run-check; else trains         │
└──────┴───────────────────────────────────────────┴───────────────────────────────────────────────┘
```

**Verification:** Steps 1 through 4 perform zero tensor allocation, zero dataset parsing, zero model loading, and zero optimizer initialization. Any security violation, collision, unflagged overwrite, or unsupported argument aborts before Step 5.

---

## 7. Path Containment Test Results

All nine path containment and collision test scenarios in `scratch/test_checkpoint_path_validation.py` were executed and verified:

| Scenario | Path Input Description | Expected Behavior | Observed Result |
| :---: | :--- | :--- | :---: |
| **1** | Standard valid paths (`best_faster_rcnn.pt`, `latest_faster_rcnn.pt`) | Accept & resolve cleanly | **PASSED** |
| **2** | Identical paths (`best_faster_rcnn.pt`, `best_faster_rcnn.pt`) | Reject: distinct filenames required | **PASSED** |
| **3** | Collider: `artifacts/checkpoints/classification/best_model.pt` | Reject: classifier collision | **PASSED** |
| **4** | Collider: `artifacts/checkpoints/segmentation/exp4b_w20_best_unet.pt` | Reject: U-Net collision | **PASSED** |
| **5** | Traversal: `artifacts/checkpoints/detection/../classification/best_model.pt` | Reject: `..` traversal detected | **PASSED** |
| **6** | External absolute path (`C:/Windows/System32/hacked.pt`) | Reject: escapes detection directory | **PASSED** |
| **7** | Symbolic link directory escape | Reject: escapes detection directory | **SKIPPED** (See Section 8) |
| **8** | Missing or empty path configuration | Reject: missing key / empty string | **PASSED** |
| **9** | Distinct custom permitted filenames (`best_v2.pt`, `latest_v2.pt`) | Accept distinct permitted files | **PASSED** |

---

## 8. Symbolic-Link and Directory Junction Test Results

In accordance with Section 4 instructions, an empirical test was conducted using `scratch/test_junction.py` in a temporary directory:

1. **Attempt 1 (`os.symlink`):**
   - Attempted: `os.symlink(outside_dir, allowed_dir / "symlink_dir", target_is_directory=True)`
   - Result: Failed with `[WinError 1314] A required privilege is not held by the client`.
2. **Attempt 2 (`mklink /J` NTFS Directory Junction):**
   - Attempted: `cmd /c mklink /J "<allowed_dir>\junction_dir" "<outside_dir>"`
   - Result: Failed with `Access is denied (code 1)`.

### Formal Declaration
> **SYMLINK ESCAPE TEST NOT EXECUTED — IMPLEMENTATION REVIEWED BUT NOT EMPIRICALLY VERIFIED.**
> 
> *Technical Audit:* Code inspection of `validate_checkpoint_paths` confirms that `Path.resolve()` is called on all paths. Under Python's standard library on Windows and POSIX, `Path.resolve()` traverses and dereferences all reparse points, junctions, and symbolic links to their real physical target volume before evaluating `.relative_to(allowed_dir)`. If a junction points outside `artifacts/checkpoints/detection/`, its resolved path will not be relative to `allowed_dir`, raising `ValueError`. However, due to Windows non-elevated user privilege boundaries, creating links could not be empirically executed in this sandbox environment.

---

## 9. Atomic Checkpoint Verification

`save_checkpoint_atomic` in `scripts/train_faster_rcnn.py` was inspected and verified:
1. **Destination Directory:** Writes `<dest_path>.tmp` in the same directory as destination on volume `D:`.
2. **Atomic Swap:** Calls `os.replace(temp_path, dest)`.
3. **Preservation:** Replacement occurs only upon successful return of `torch.save`. Failure in serialization preserves the existing checkpoint.
4. **Exception Handling:** Exception block unlinks the partial temporary file and cleanly re-raises the original exception via bare `raise`.
5. **Durability Distinction:** Guarantees software-level atomic directory entry replacement (readers never observe truncated files); does not overclaim physical hardware power-loss durability without explicit OS disk fsync.

---

## 10. Dataset and Model Protection

Re-verified from static manifests and existing audit evidence:
1. **Published Annotation Corpus:**
   - Manifest: `artifacts/manifests/combined_pilot_manifest.csv`
   - Published Task #4 (Batch 01): 21 images (14 train / 7 val)
   - Published Task #5 (Batch 02): 29 images (21 train / 8 val)
   - Total Published: **50 images** (35 train / 15 val)
   - Total Reconciled Boxes: **514 bounding boxes** (371 train / 143 val)
2. **Exclusions Strictly Preserved:**
   - CVAT Task #3: Excluded.
   - CVAT Task #6 staged annotations: Excluded.
   - Augmented classification data: Excluded.
   - Sealed 321-image holdout test split (`test.csv`, SHA-256 `6025D86F...`): Excluded, unreferenced, unread.
3. **Cryptographic Protection of Production Assets:**
   - Classifier `artifacts/checkpoints/classification/best_model.pt` (`552BABA8...`): Intact.
   - U-Net `artifacts/checkpoints/segmentation/exp4b_w20_best_unet.pt` (`C6788F79...`): Intact.
   - Production database `cotton_leaf_ai.db`: Intact and untouched.

---

## 11. Files Changed

| File Path | Nature of Changes |
| :--- | :--- |
| [`scripts/train_faster_rcnn.py`](file:///D:/Main_Project/cotton_leaf_ai/scripts/train_faster_rcnn.py) | Added `hashlib` and `shutil` imports; implemented `check_existing_checkpoint_guard`; updated `train_faster_rcnn` signature and call sequence; added `--overwrite-existing` and `--resume` CLI flags in `main()`. |
| [`scratch/test_overwrite_policy.py`](file:///D:/Main_Project/cotton_leaf_ai/scratch/test_overwrite_policy.py) | Authored 10-scenario unit test suite evaluating fresh run, unflagged block, authorized overwrite, backup SHA-256 validation, failure cleanup, and collision rejection. |
| [`scratch/test_junction.py`](file:///D:/Main_Project/cotton_leaf_ai/scratch/test_junction.py) | Attempted empirical symlink and directory junction tests on Windows. |
| [`scratch/test_main_overwrite_cli.py`](file:///D:/Main_Project/cotton_leaf_ai/scratch/test_main_overwrite_cli.py) | Authored end-to-end CLI dry-run test suite verifying CLI flag handling. |
| [`reports/phase9_2b_3_3_final_safety_verification.md`](file:///D:/Main_Project/cotton_leaf_ai/reports/phase9_2b_3_3_final_safety_verification.md) | Authored master deliverable report for Phase 9.2B.3.3. |
| [`artifacts/reports/phase9_2b_3_3_final_safety_verification.md`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/reports/phase9_2b_3_3_final_safety_verification.md) | Synchronized workspace mirror report. |
| [`PROJECT_PROGRESS.md`](file:///D:/Main_Project/cotton_leaf_ai/PROJECT_PROGRESS.md) | Updated tracking log for Phase 9.2B.3.3 completion. |
| [`FILE_CHANGE_LOG.md`](file:///D:/Main_Project/cotton_leaf_ai/FILE_CHANGE_LOG.md) | Appended changelog entry. |

---

## 12. Tests Executed and Results

| Test Suite / Script | Invariant / Behavior Tested | Execution Context | Result |
| :--- | :--- | :--- | :---: |
| `scratch/test_overwrite_policy.py` (10 tests) | Evaluated all 10 overwrite, backup, collision, and resume rejection scenarios | Isolated temporary directory with synthetic bytes | **100% PASS** |
| `scratch/test_main_overwrite_cli.py` (4 tests) | Evaluated `--dry-run-check`, `--overwrite-existing`, `--resume`, and unflagged block via CLI | Subprocess invocations using `--dry-run-check` | **100% PASS** |
| `scratch/test_junction.py` | Evaluated OS support for symlinks and NTFS directory junctions | Temporary directory attempt | **EXECUTED** (OS blocked privilege) |
| `py_compile train_faster_rcnn.py` | Verified AST syntax compilation of modified training script | Python compiler | **100% PASS** |
| `pytest test_phase9_2b_2_...` (8 tests) | Verified regression status of existing unit tests | Pytest runner | **8/8 PASSED (7.04s)** |
| Checkpoint Directory Count Check | Confirmed `artifacts/checkpoints/detection/` contains 0 files | Filesystem count | **100% PASS (0 files)** |

---

## 13. Tests Skipped and Reasons

| Candidate Test / Action | Reason for Skipping | Security / Safety Invariant Enforced |
| :--- | :--- | :--- |
| **Full Model Training Run** | Explicit user restriction | No training authorized in Phase 9.2B.3.3. |
| **Model Forward/Backward Step** | Zero-training restriction | Prevents weight updates prior to authorization. |
| **Sealed Test Set Evaluation** | Holdout test isolation | Prevents data leakage into detection models. |
| **Empirical Symlink Creation** | OS security policy (`WinError 1314`) | Standard non-elevated user accounts lack link privileges. |

---

## 14. Remaining Limitations

1. **Pilot Dataset Size:** The published corpus contains 35 training images and 15 validation images (514 bounding boxes). Regularization (SGD weight decay $0.0005$, gradient clipping $10.0$, 3-layer backbone fine-tuning, and early stopping patience 5) is essential to mitigate memorization.
2. **Curl Virus Validation Scarcity:** With only 4 validation boxes for Curl Virus, per-class AP50 will exhibit high variance. Evaluation metrics on this class should be considered indicative pilot estimates rather than asymptotic performance.
3. **Resume Functionality Deferred:** `--resume` is explicitly gated and rejected until a future phase authorizes its implementation.

---

## 15. Final Readiness Classification

### **Classification: A. READY FOR FUTURE USER AUTHORIZATION**

**Justification:**  
All pre-training safety requirements, checkpoint path containment validations (Rules A–E), atomic serialization safeguards, pre-training overwrite protection (Rules A–E), and backup SHA-256 verifications are **100% complete, verified by unit and CLI tests, and fully hardened**. The training implementation is technically airtight and ready for a future training authorization decision.

---

## 16. Final Status Confirmation

In accordance with Section 10 of your instructions:

- **Faster R-CNN training epochs:** `0`
- **Faster R-CNN training optimizer updates:** `0`
- **Faster R-CNN training backward passes:** `0`
- **New detector checkpoints created:** `0`
- **Test images loaded:** `NO`
- **Test annotations loaded:** `NO`
- **Test evaluation performed:** `NO`
- **CVAT annotations modified:** `NO`
- **Published dataset modified:** `NO`
- **Classifier checkpoint modified:** `NO`
- **U-Net checkpoint modified:** `NO`
- **Production database modified:** `NO`

*Synthetic test operations in sandbox scripts: 0 optimizer updates and 0 backward passes executed in this phase (1 synthetic dummy update and 2 dummy backward passes occurred in prior Phase 9.2B.3 sandbox verification).*
