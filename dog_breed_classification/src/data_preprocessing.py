"""
Data Preprocessing, Splitting, and Augmentation Module
Phase 2 - Dog Breed Classification Using MobileNetV2-Based Transfer Learning

This module provides:
1. Dynamic dataset loading and structured DataFrame creation (image_path, breed, label)
2. Consistent alphabetical class name to numerical ID mapping (0 to NUM_CLASSES - 1)
3. Stratified Train / Validation / Test split (80% / 10% / 10%) with strict data leakage checks
4. Data augmentation pipeline (RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast)
5. Dynamic 224x224 RGB image loading and official MobileNetV2 preprocess_input() scaling
6. Optimized tf.data.Dataset pipelines (shuffle, batch, augment, prefetch)
7. Class imbalance analysis and balanced class weight calculation
8. Visualization of data augmentation examples across multiple dog breeds
9. Single-image preprocessing utility ready for Phase 3/4 inference
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
TARGET_IMAGE_SIZE = (224, 224)
TARGET_CHANNELS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_STATE = 42


def resolve_paths(
    data_dir_arg: Optional[str] = None,
    labels_csv_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None
) -> Tuple[Path, Path, Path]:
    """
    Locates dataset directories and labels.csv without hardcoding absolute paths.
    Uses relative search candidates and environment variables.
    """
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent

    # Candidate data dirs
    cand_dirs = [
        data_dir_arg,
        os.environ.get("DATASET_DIR"),
        proj_root / "dataset" / "train",
        proj_root / "dataset",
        Path("dog_breed_classification/dataset/train"),
        Path("dog_breed_classification/dataset"),
        Path("dataset/train"),
        Path("dataset"),
        Path("../dataset/train"),
        Path("../dataset"),
        Path("train"),
        Path("../train")
    ]

    # Candidate labels
    cand_labels = [
        labels_csv_arg,
        os.environ.get("LABELS_CSV"),
        proj_root / "dataset" / "labels.csv",
        proj_root / "labels.csv",
        Path("dog_breed_classification/dataset/labels.csv"),
        Path("dataset/labels.csv"),
        Path("labels.csv"),
        Path("../dataset/labels.csv"),
        Path("../labels.csv")
    ]

    resolved_data_dir = None
    for cand in cand_dirs:
        if cand is not None:
            p = Path(cand)
            if p.is_dir():
                sub_train = p / "train"
                if sub_train.is_dir():
                    files = [f for f in os.listdir(sub_train) if (sub_train / f).is_file()]
                    if files:
                        resolved_data_dir = sub_train
                        break
                files = [f for f in os.listdir(p) if (p / f).is_file() and not f.endswith((".csv", ".md", ".txt"))]
                if files:
                    resolved_data_dir = p
                    break

    resolved_labels_csv = None
    for cand in cand_labels:
        if cand is not None:
            p = Path(cand)
            if p.is_file():
                resolved_labels_csv = p
                break

    if not resolved_data_dir:
        raise FileNotFoundError("Could not locate training images folder. Please specify via data_dir_arg.")
    if not resolved_labels_csv:
        raise FileNotFoundError("Could not locate labels.csv. Please specify via labels_csv_arg.")

    if output_dir_arg:
        resolved_output_dir = Path(output_dir_arg)
    else:
        if (proj_root / "outputs").is_dir() or proj_root.exists():
            resolved_output_dir = proj_root / "outputs"
        else:
            resolved_output_dir = Path("outputs")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return Path(resolved_data_dir), Path(resolved_labels_csv), Path(resolved_output_dir)


def load_labels(labels_path: Union[str, Path]) -> pd.DataFrame:
    """Loads and validates labels.csv."""
    df = pd.read_csv(labels_path)
    df.columns = df.columns.str.strip()
    if "id" not in df.columns or "breed" not in df.columns:
        raise ValueError(f"labels.csv must contain 'id' and 'breed' columns. Found: {list(df.columns)}")
    df["id"] = df["id"].astype(str).str.strip()
    df["breed"] = df["breed"].astype(str).str.strip()
    return df


def create_dataframe(
    data_dir: Union[str, Path],
    labels_path: Union[str, Path]
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str], List[str]]:
    """
    Creates structured DataFrame with columns: ['image_path', 'breed', 'label', 'id'].
    Maps breed names to consistent numerical class IDs (0 to NUM_CLASSES - 1).
    """
    data_dir = Path(data_dir)
    df_labels = load_labels(labels_path)

    # Scan available files in image directory to detect extensions dynamically
    image_files = [f for f in os.listdir(data_dir) if (data_dir / f).is_file()]
    stem_to_filename = {Path(f).stem: f for f in image_files}

    # Match IDs to actual image filenames
    df_labels["filename"] = df_labels["id"].map(stem_to_filename)
    missing_count = df_labels["filename"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Found {missing_count} IDs in labels.csv without corresponding image files!")

    df_matched = df_labels.dropna(subset=["filename"]).copy()
    df_matched["image_path"] = df_matched["filename"].apply(lambda f: str((data_dir / f).resolve()))

    # Consistent alphabetical sorting for class names
    class_names = sorted(df_matched["breed"].unique())
    num_classes = len(class_names)
    logger.info(f"Automatically detected {num_classes} unique dog breeds.")

    class_to_id = {breed: idx for idx, breed in enumerate(class_names)}
    id_to_class = {idx: breed for idx, breed in enumerate(class_names)}

    df_matched["label"] = df_matched["breed"].map(class_to_id)

    # Reorder columns
    df_final = df_matched[["image_path", "breed", "label", "id", "filename"]].copy()
    logger.info(f"Created DataFrame with {len(df_final)} verified samples.")
    return df_final, class_to_id, id_to_class, class_names


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/val/test split (default 80% / 10% / 10%).
    Ensures zero data leakage between splits.
    """
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split proportions must sum to 1.0 (got {total})")

    # Step 1: Stratified split into train (80%) and temp (20%)
    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=df["label"],
        random_state=random_state
    )

    # Step 2: Stratified split of temp into validation (50% of 20% = 10%) and test (10%)
    test_ratio_of_temp = test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio_of_temp,
        stratify=temp_df["label"],
        random_state=random_state
    )

    # Strict Data Leakage Verification
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    leakage_train_val = train_paths.intersection(val_paths)
    leakage_train_test = train_paths.intersection(test_paths)
    leakage_val_test = val_paths.intersection(test_paths)

    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! "
            f"Train/Val overlap: {len(leakage_train_val)}, "
            f"Train/Test overlap: {len(leakage_train_test)}, "
            f"Val/Test overlap: {len(leakage_val_test)}"
        )

    logger.info(
        f"Stratified Dataset Split: "
        f"Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
        f"Val={len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
        f"Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)"
    )
    logger.info("Data leakage verification: PASSED (0 overlapping images across all splits).")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights_dict(train_df: pd.DataFrame, num_classes: int) -> Dict[int, float]:
    """
    Computes balanced class weights for the training split.
    w_j = N_samples / (N_classes * N_samples_j)
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["label"].values
    )
    class_weights = {int(cls_id): float(round(w, 5)) for cls_id, w in zip(classes, weights)}
    return class_weights


def create_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Constructs reasonable data augmentation pipeline for training images only.
    Avoids extreme distortions that could alter dog breed characteristics.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", name="random_flip_h"),
        tf.keras.layers.RandomRotation(0.10, fill_mode="reflect", name="random_rotation"),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", name="random_zoom"),
        tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="reflect", name="random_translation"),
        tf.keras.layers.RandomContrast(0.10, name="random_contrast")
    ], name="mobilenetv2_data_augmentation")


def _load_image_raw(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads image, decodes JPEG as RGB, resizes to (224, 224), and casts to float32."""
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=TARGET_CHANNELS)
    img = tf.image.resize(img, TARGET_IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img, label


def create_dataset(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    is_training: bool = False,
    augment_model: Optional[tf.keras.Sequential] = None,
    one_hot: bool = False,
    num_classes: Optional[int] = None,
    shuffle_buffer: int = 1000
) -> tf.data.Dataset:
    """
    Creates an optimized tf.data.Dataset pipeline.
    - Resizes to (224, 224, 3)
    - Augments ONLY if is_training=True
    - Scales using official MobileNetV2 preprocess_input() -> [-1, 1]
    - Batches and prefetches using AUTOTUNE
    """
    image_paths = df["image_path"].values
    labels = df["label"].values

    if one_hot:
        if num_classes is None:
            num_classes = df["label"].nunique()
        labels_tensor = tf.one_hot(labels, depth=num_classes)
    else:
        labels_tensor = tf.constant(labels, dtype=tf.int32)

    paths_tensor = tf.constant(image_paths, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), shuffle_buffer), seed=DEFAULT_RANDOM_STATE, reshuffle_each_iteration=True)

    # Load and resize images
    ds = ds.map(_load_image_raw, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch images
    ds = ds.batch(batch_size, drop_remainder=False)

    # Training: Apply augmentation on batch, then preprocess_input
    if is_training and augment_model is not None:
        def augment_and_preprocess(batch_images, batch_labels):
            aug_imgs = augment_model(batch_images, training=True)
            prep_imgs = preprocess_input(aug_imgs)
            return prep_imgs, batch_labels

        ds = ds.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Validation / Test: preprocess_input directly without augmentation
        def preprocess_val(batch_images, batch_labels):
            prep_imgs = preprocess_input(batch_images)
            return prep_imgs, batch_labels

        ds = ds.map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_training_dataset(
    train_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment_model: Optional[tf.keras.Sequential] = None,
    one_hot: bool = False,
    num_classes: Optional[int] = None
) -> tf.data.Dataset:
    """Helper to create training dataset with augmentation and shuffling."""
    if augment_model is None:
        augment_model = create_augmentation_pipeline()
    return create_dataset(
        df=train_df,
        batch_size=batch_size,
        is_training=True,
        augment_model=augment_model,
        one_hot=one_hot,
        num_classes=num_classes
    )


def create_validation_dataset(
    val_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    one_hot: bool = False,
    num_classes: Optional[int] = None
) -> tf.data.Dataset:
    """Helper to create validation dataset without augmentation."""
    return create_dataset(
        df=val_df,
        batch_size=batch_size,
        is_training=False,
        augment_model=None,
        one_hot=one_hot,
        num_classes=num_classes
    )


def create_test_dataset(
    test_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    one_hot: bool = False,
    num_classes: Optional[int] = None
) -> tf.data.Dataset:
    """Helper to create testing dataset without augmentation."""
    return create_dataset(
        df=test_df,
        batch_size=batch_size,
        is_training=False,
        augment_model=None,
        one_hot=one_hot,
        num_classes=num_classes
    )


def preprocess_single_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = TARGET_IMAGE_SIZE
) -> np.ndarray:
    """
    Reusable inference utility: prepares a single arbitrary dog image for MobileNetV2 prediction.
    Outputs: Tensor of shape (1, 224, 224, 3) in range [-1.0, 1.0].
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.shape[-1] == 4:
            image_input = image_input[:, :, :3]
        img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32)

    # Batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # MobileNetV2 official preprocessing
    preprocessed_tensor = preprocess_input(img_batch)
    return preprocessed_tensor


def visualize_augmentation(
    train_df: pd.DataFrame,
    augment_model: tf.keras.Sequential,
    output_path: Union[str, Path],
    num_samples: int = 4
) -> None:
    """
    Visualizes Original Image vs 4 Augmented Variants across multiple dog breeds.
    Saves output to outputs/augmentation_examples.png.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick samples from distinct dog breeds
    unique_breeds = train_df["breed"].unique()
    selected_breeds = np.random.choice(unique_breeds, size=min(num_samples, len(unique_breeds)), replace=False)

    sample_rows = [train_df[train_df["breed"] == b].iloc[0] for b in selected_breeds]

    fig, axes = plt.subplots(num_samples, 5, figsize=(18, 3.6 * num_samples))

    for row_idx, row in enumerate(sample_rows):
        img_path = row["image_path"]
        breed_name = row["breed"].replace("_", " ").title()

        # Load raw original image resized to 224x224
        with Image.open(img_path) as raw_img:
            orig_img = raw_img.convert("RGB").resize(TARGET_IMAGE_SIZE, Image.Resampling.BILINEAR)
            orig_np = np.array(orig_img, dtype=np.float32)

        # Plot Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig_np.astype(np.uint8))
        ax_orig.set_title(f"Original\n({breed_name})", fontsize=11, fontweight="bold", pad=4)
        ax_orig.axis("off")

        # Generate 4 augmented variants
        orig_batch = tf.expand_dims(orig_np, axis=0)
        for aug_idx in range(1, 5):
            aug_img_tensor = augment_model(orig_batch, training=True)
            aug_np = aug_img_tensor[0].numpy()
            aug_display = np.clip(aug_np, 0, 255).astype(np.uint8)

            ax_aug = axes[row_idx, aug_idx]
            ax_aug.imshow(aug_display)
            ax_aug.set_title(f"Augmented #{aug_idx}", fontsize=10, pad=4)
            ax_aug.axis("off")

    plt.suptitle("MobileNetV2 Data Augmentation Examples (Training Set Only)", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved augmentation visualization to: {output_path}")


def verify_data_pipeline(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    expected_num_classes: int,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dict[str, Union[bool, str, List[int]]]:
    """
    Verifies that batches load correctly with expected shape, range [-1, 1], and no NaNs/Infs.
    """
    logger.info("Verifying data pipelines across all splits...")

    results = {}
    splits = [("train", train_ds), ("val", val_ds), ("test", test_ds)]

    for name, ds in splits:
        for batch_images, batch_labels in ds.take(1):
            img_shape = list(batch_images.shape)
            lbl_shape = list(batch_labels.shape)
            min_val = float(tf.reduce_min(batch_images).numpy())
            max_val = float(tf.reduce_max(batch_images).numpy())
            has_nan = bool(tf.reduce_any(tf.math.is_nan(batch_images)).numpy())
            has_inf = bool(tf.reduce_any(tf.math.is_inf(batch_images)).numpy())

            logger.info(
                f"[{name.upper()}] Batch images shape: {img_shape}, "
                f"labels shape: {lbl_shape}, "
                f"value range: [{min_val:.3f}, {max_val:.3f}], "
                f"has_nan: {has_nan}, has_inf: {has_inf}"
            )

            # Verification assertions
            assert img_shape[1:] == [224, 224, 3], f"Unexpected image shape: {img_shape}"
            assert not has_nan, f"NaN values detected in {name} batch!"
            assert not has_inf, f"Inf values detected in {name} batch!"
            assert -1.05 <= min_val <= 1.05 and -1.05 <= max_val <= 1.05, f"Values out of MobileNetV2 range: [{min_val}, {max_val}]"

            results[f"{name}_shape"] = img_shape
            results[f"{name}_min"] = min_val
            results[f"{name}_max"] = max_val
            results[f"{name}_valid"] = True

    return results


def run_phase_2(
    data_dir_arg: Optional[str] = None,
    labels_csv_arg: Optional[str] = None,
    output_dir_arg: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Dict[str, Union[int, str, float, bool]]:
    """
    Executes full Phase 2 preprocessing, splitting, augmentation, and verification.
    """
    data_dir, labels_csv, output_dir = resolve_paths(data_dir_arg, labels_csv_arg, output_dir_arg)

    logger.info("=== PHASE 2: DATA PREPROCESSING & AUGMENTATION ===")
    logger.info(f"Image Directory:    {data_dir}")
    logger.info(f"Labels CSV:         {labels_csv}")
    logger.info(f"Outputs Directory:  {output_dir}")

    # 1. Create DataFrame
    df, class_to_id, id_to_class, class_names = create_dataframe(data_dir, labels_csv)
    num_classes = len(class_names)
    total_images = len(df)

    # Save class names mapping
    class_names_payload = {
        "num_classes": num_classes,
        "class_names": class_names,
        "class_to_id": class_to_id,
        "id_to_class": {str(k): v for k, v in id_to_class.items()}
    }
    class_names_path = output_dir / "class_names.json"
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names_payload, f, indent=2)
    logger.info(f"Saved class mappings to: {class_names_path}")

    # 2. Stratified Dataset Split (80% / 10% / 10%)
    train_df, val_df, test_df = split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1)

    # Save split CSVs for exact reproducibility in Phase 3
    train_csv_path = output_dir / "train_split.csv"
    val_csv_path = output_dir / "val_split.csv"
    test_csv_path = output_dir / "test_split.csv"
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    logger.info("Saved split CSVs (train_split.csv, val_split.csv, test_split.csv).")

    # 3. Class Imbalance Analysis & Class Weights
    class_weights = compute_class_weights_dict(train_df, num_classes)
    weight_min = min(class_weights.values())
    weight_max = max(class_weights.values())
    weight_ratio = weight_max / weight_min
    is_imbalanced = bool(weight_ratio > 1.5)

    class_weights_path = output_dir / "class_weights.json"
    class_weights_payload = {
        "is_imbalanced": is_imbalanced,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "imbalance_ratio": round(weight_ratio, 2),
        "class_weights": {str(k): v for k, v in class_weights.items()}
    }
    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights_payload, f, indent=2)
    logger.info(f"Saved class weights to: {class_weights_path}")

    # 4. Augmentation Pipeline & Visualization
    augment_model = create_augmentation_pipeline()
    aug_vis_path = output_dir / "augmentation_examples.png"
    visualize_augmentation(train_df, augment_model, aug_vis_path, num_samples=4)

    # 5. Build and Verify tf.data.Dataset pipelines
    logger.info(f"Building tf.data pipelines with batch_size={batch_size}...")
    train_ds = create_training_dataset(train_df, batch_size=batch_size, augment_model=augment_model)
    val_ds = create_validation_dataset(val_df, batch_size=batch_size)
    test_ds = create_test_dataset(test_df, batch_size=batch_size)

    verification_results = verify_data_pipeline(train_ds, val_ds, test_ds, num_classes, batch_size=batch_size)

    # 6. Save Split & Preprocessing Summary
    split_summary = {
        "total_images": total_images,
        "num_classes": num_classes,
        "splits": {
            "train": {
                "count": len(train_df),
                "percentage": round(len(train_df) / total_images * 100, 2),
                "num_classes_represented": train_df["label"].nunique()
            },
            "validation": {
                "count": len(val_df),
                "percentage": round(len(val_df) / total_images * 100, 2),
                "num_classes_represented": val_df["label"].nunique()
            },
            "test": {
                "count": len(test_df),
                "percentage": round(len(test_df) / total_images * 100, 2),
                "num_classes_represented": test_df["label"].nunique()
            }
        },
        "image_size": list(TARGET_IMAGE_SIZE) + [TARGET_CHANNELS],
        "batch_size": batch_size,
        "augmentation_techniques": [
            "RandomFlip(horizontal)",
            "RandomRotation(0.10)",
            "RandomZoom(0.10)",
            "RandomTranslation(0.10, 0.10)",
            "RandomContrast(0.10)"
        ],
        "preprocessing": "tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1])",
        "class_imbalance": {
            "imbalanced": is_imbalanced,
            "weight_min": weight_min,
            "weight_max": weight_max,
            "ratio": round(weight_ratio, 2)
        },
        "data_leakage_detected": False,
        "pipeline_verification": verification_results,
        "status": "READY FOR MODEL TRAINING"
    }

    summary_path = output_dir / "dataset_split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)
    logger.info(f"Saved dataset split summary to: {summary_path}")

    # 7. Print Structured Phase 2 Summary Block
    imbalance_str = "Yes (mild, class weights computed and saved)" if is_imbalanced else "No"
    print("\n" + "=" * 40)
    print("DATASET PREPROCESSING SUMMARY")
    print("=" * 40)
    print(f"\nTotal images:        {total_images}")
    print(f"Number of breeds:    {num_classes}")
    print(f"\nTraining images:     {len(train_df)}")
    print(f"Validation images:   {len(val_df)}")
    print(f"Testing images:      {len(test_df)}")
    print(f"\nImage size:\n{TARGET_IMAGE_SIZE[0]} × {TARGET_IMAGE_SIZE[1]} × {TARGET_CHANNELS}")
    print("\nAugmentation:\nEnabled for training only")
    print("\nMobileNetV2 preprocessing:\nEnabled")
    print(f"\nClass imbalance:\n{imbalance_str}")
    print("\nData leakage:\nNone detected")
    print("\nStatus:\nREADY FOR MODEL TRAINING")
    print("=" * 40 + "\n")

    return split_summary


if __name__ == "__main__":
    run_phase_2()
