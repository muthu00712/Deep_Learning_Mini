"""
Dataset Analysis and Verification Script
Phase 1 - Dog Breed Classification Using MobileNetV2-Based Transfer Learning

This script performs:
1. Environment and hardware availability verification (Python, TensorFlow, CPU/GPU)
2. Configurable dataset discovery and dynamic extension detection
3. Complete image-to-label mapping and validation (id -> image filename -> breed)
4. Full validation across all images (missing files, corrupt images, duplicates, dimensions)
5. Comprehensive statistical analysis of classes and image dimensions
6. Visualizations and report generation in the outputs/ directory
"""

import os
import sys
import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for reliable server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Verify Python, TensorFlow, and hardware devices."""
    env_info = {
        "python_version": sys.version.split()[0],
        "tensorflow_version": "Not installed",
        "keras_version": "Not installed",
        "cpu_available": True,
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_devices": []
    }

    try:
        import tensorflow as tf
        env_info["tensorflow_version"] = tf.__version__
        try:
            import keras
            env_info["keras_version"] = keras.__version__
        except Exception:
            env_info["keras_version"] = getattr(tf, "keras", {}).get("__version__", "Bundled with TF")

        gpus = tf.config.list_physical_devices("GPU")
        env_info["gpu_count"] = len(gpus)
        env_info["gpu_available"] = len(gpus) > 0
        env_info["gpu_devices"] = [gpu.name for gpu in gpus]
    except Exception as e:
        logger.warning(f"Error checking TensorFlow: {e}")

    logger.info("=== ENVIRONMENT & HARDWARE STATUS ===")
    logger.info(f"Python Version:       {env_info['python_version']}")
    logger.info(f"TensorFlow Version:   {env_info['tensorflow_version']}")
    logger.info(f"CPU Availability:     {env_info['cpu_available']} (Primary execution target)")
    logger.info(f"GPU Availability:     {env_info['gpu_available']} (GPUs detected: {env_info['gpu_count']})")
    if env_info["gpu_devices"]:
        logger.info(f"GPU Devices:          {env_info['gpu_devices']}")
    logger.info("=====================================")
    return env_info


def find_dataset(data_dir_arg=None, labels_csv_arg=None):
    """
    Locates the dataset directory and labels.csv without hardcoding absolute paths.
    Uses argument inputs, environment variables, or relative candidate search paths.
    """
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent

    candidate_data_dirs = [
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

    candidate_labels_files = [
        labels_csv_arg,
        os.environ.get("LABELS_CSV"),
        proj_root / "dataset" / "labels.csv",
        proj_root / "labels.csv",
        Path("dog_breed_classification/dataset/labels.csv"),
        Path("dog_breed_classification/labels.csv"),
        Path("dataset/labels.csv"),
        Path("labels.csv"),
        Path("../dataset/labels.csv"),
        Path("../labels.csv")
    ]

    resolved_data_dir = None
    for cand in candidate_data_dirs:
        if cand is not None:
            cand_p = Path(cand)
            if cand_p.is_dir():
                # If this directory has a 'train' subdirectory with images, prefer that
                train_sub = cand_p / "train"
                if train_sub.is_dir():
                    sub_files = [f for f in os.listdir(train_sub) if (train_sub / f).is_file()]
                    if sub_files:
                        resolved_data_dir = train_sub
                        break
                files = [f for f in os.listdir(cand_p) if (cand_p / f).is_file() and not f.endswith((".csv", ".md", ".txt"))]
                if files:
                    resolved_data_dir = cand_p
                    break

    resolved_labels_csv = None
    for cand in candidate_labels_files:
        if cand is not None:
            cand_p = Path(cand)
            if cand_p.is_file():
                resolved_labels_csv = cand_p
                break

    if not resolved_data_dir:
        raise FileNotFoundError("Could not locate training images folder. Please specify via --data-dir.")
    if not resolved_labels_csv:
        raise FileNotFoundError("Could not locate labels.csv. Please specify via --labels.")

    return Path(resolved_data_dir), Path(resolved_labels_csv)


def inspect_single_image(file_path):
    """
    Validates a single image file and extracts metadata.
    Returns a dictionary of status and image properties.
    """
    result = {
        "filename": file_path.name,
        "is_valid": False,
        "width": 0,
        "height": 0,
        "mode": "Unknown",
        "format": "Unknown",
        "is_corrupt": False,
        "is_extremely_small": False,
        "error_msg": None
    }

    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            result["is_corrupt"] = True
            result["error_msg"] = "Empty file (0 bytes)"
            return result

        with Image.open(file_path) as img:
            result["width"], result["height"] = img.size
            result["mode"] = img.mode
            result["format"] = img.format

            # Test reading image integrity
            img.verify()

        # Also confirm decompression by re-opening and checking a slice or thumbnail
        with Image.open(file_path) as img:
            img.load()

        if result["width"] < 32 or result["height"] < 32:
            result["is_extremely_small"] = True

        result["is_valid"] = True
    except Exception as e:
        result["is_corrupt"] = True
        result["error_msg"] = str(e)

    return result


def analyze_dataset(data_dir, labels_csv_path, output_dir, sample_size=16, num_workers=8):
    """
    Executes full dataset verification, statistical analysis, and visualization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading labels from: {labels_csv_path}")
    df_labels = pd.read_csv(labels_csv_path)

    # Clean column names
    df_labels.columns = df_labels.columns.str.strip()

    if "id" not in df_labels.columns or "breed" not in df_labels.columns:
        raise ValueError(f"labels.csv must contain 'id' and 'breed' columns. Found: {list(df_labels.columns)}")

    # Ensure string types
    df_labels["id"] = df_labels["id"].astype(str).str.strip()
    df_labels["breed"] = df_labels["breed"].astype(str).str.strip()

    logger.info(f"Scanning image directory: {data_dir}")
    available_files = [f for f in os.listdir(data_dir) if os.path.isfile(data_dir / f)]
    logger.info(f"Total files found in image directory: {len(available_files)}")

    # Build filename and extension mapping dynamically (do not assume .jpg)
    stem_to_file = {}
    extension_counter = Counter()

    for fname in available_files:
        f_path = Path(fname)
        ext = f_path.suffix.lower()
        extension_counter[ext] += 1
        stem = f_path.stem
        # Map ID stem to full filename
        stem_to_file[stem] = fname

    logger.info(f"Detected file extensions: {dict(extension_counter)}")

    # Check duplicates in labels.csv
    duplicate_ids_count = df_labels["id"].duplicated().sum()
    logger.info(f"Duplicate IDs in labels.csv: {duplicate_ids_count}")

    # Check labels without images and images without labels
    label_ids = set(df_labels["id"])
    image_stems = set(stem_to_file.keys())

    labels_without_images = label_ids - image_stems
    images_without_labels = image_stems - label_ids

    logger.info(f"Labels without images: {len(labels_without_images)}")
    logger.info(f"Images without labels: {len(images_without_labels)}")

    # Build verified mapping
    df_labels["filename"] = df_labels["id"].map(stem_to_file)
    df_matched = df_labels.dropna(subset=["filename"]).copy()

    # Verify each image file integrity in parallel
    logger.info(f"Validating {len(df_matched)} images for corruption, dimensions, and readability...")
    matched_paths = [data_dir / row["filename"] for _, row in df_matched.iterrows()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        image_inspection_results = list(executor.map(inspect_single_image, matched_paths))

    df_inspect = pd.DataFrame(image_inspection_results)
    df_merged = pd.concat([df_matched.reset_index(drop=True), df_inspect.drop(columns=["filename"])], axis=1)

    # Anomaly summary
    corrupted_count = df_merged["is_corrupt"].sum()
    extremely_small_count = df_merged["is_extremely_small"].sum()
    unreadable_count = (~df_merged["is_valid"]).sum()
    color_mode_counts = df_merged["mode"].value_counts().to_dict()
    format_counts = df_merged["format"].value_counts().to_dict()

    logger.info(f"Corrupted images:       {corrupted_count}")
    logger.info(f"Unreadable images:      {unreadable_count}")
    logger.info(f"Extremely small (<32):  {extremely_small_count}")
    logger.info(f"Color modes breakdown:  {color_mode_counts}")
    logger.info(f"Image formats breakdown:{format_counts}")

    # Class statistics
    total_images = len(available_files)
    total_training_images = len(df_matched)
    unique_breeds = sorted(df_labels["breed"].unique())
    num_breeds = len(unique_breeds)

    breed_counts = df_matched["breed"].value_counts()
    most_common_breed = breed_counts.index[0]
    most_common_count = int(breed_counts.iloc[0])
    least_common_breed = breed_counts.index[-1]
    least_common_count = int(breed_counts.iloc[-1])
    mean_count = float(breed_counts.mean())
    median_count = float(breed_counts.median())
    std_count = float(breed_counts.std())
    imbalance_ratio = most_common_count / least_common_count if least_common_count > 0 else float("inf")

    # Balance evaluation
    if imbalance_ratio < 1.5:
        balance_verdict = "Well-balanced (all classes have nearly uniform representation)"
    elif imbalance_ratio < 3.0:
        balance_verdict = "Moderately balanced (mild variance across classes, typical for Kaggle dog breeds)"
    else:
        balance_verdict = "Significantly imbalanced (substantial disparity between classes)"

    # Image dimension statistics
    valid_dims = df_merged[df_merged["is_valid"]]
    min_width = int(valid_dims["width"].min())
    max_width = int(valid_dims["width"].max())
    mean_width = float(valid_dims["width"].mean())
    median_width = float(valid_dims["width"].median())

    min_height = int(valid_dims["height"].min())
    max_height = int(valid_dims["height"].max())
    mean_height = float(valid_dims["height"].mean())
    median_height = float(valid_dims["height"].median())

    # Mode of dimensions
    dim_pairs = list(zip(valid_dims["width"], valid_dims["height"]))
    most_common_dim, most_common_dim_freq = Counter(dim_pairs).most_common(1)[0]

    rgb_count = int(color_mode_counts.get("RGB", 0))
    grayscale_count = int(color_mode_counts.get("L", 0)) + int(color_mode_counts.get("1", 0))

    # Save breed names list
    with open(output_dir / "breed_names.txt", "w", encoding="utf-8") as f:
        for b in unique_breeds:
            f.write(f"{b}\n")

    # Save breed distribution table
    breed_dist_df = pd.DataFrame({
        "breed": breed_counts.index,
        "image_count": breed_counts.values,
        "percentage": (breed_counts.values / total_training_images * 100).round(3)
    })
    breed_dist_df.to_csv(output_dir / "breed_distribution.csv", index=False)

    # ----------------- Visualizations ----------------- #
    logger.info("Generating visualizations...")

    # 1. Sample training images with labels from labels.csv
    np.random.seed(42)
    sample_indices = np.random.choice(len(df_matched), size=min(sample_size, len(df_matched)), replace=False)
    sample_rows = df_matched.iloc[sample_indices]

    grid_cols = 4
    grid_rows = int(np.ceil(len(sample_rows) / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 4 * grid_rows))
    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(sample_rows.iterrows()):
        ax = axes[i]
        img_path = data_dir / row["filename"]
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                ax.imshow(img_rgb)
                w, h = img.size
                formatted_breed = row["breed"].replace("_", " ").title()
                ax.set_title(f"Breed: {formatted_breed}\nID: {row['id'][:10]}... ({w}x{h})", fontsize=10, pad=4)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        ax.axis("off")

    # Turn off any remaining unused axes
    for j in range(len(sample_rows), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Training Images with Verified Breed Labels (Phase 1)", fontsize=16, y=0.99)
    plt.tight_layout()
    sample_plot_path = output_dir / "sample_training_images.png"
    plt.savefig(sample_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample image grid to: {sample_plot_path}")

    # 2. Class Distribution Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Top 15 and Bottom 15 breeds
    top_15 = breed_counts.head(15)
    bottom_15 = breed_counts.tail(15)
    combo_breeds = pd.concat([top_15, bottom_15])
    colors = ["#1f77b4"] * 15 + ["#ff7f0e"] * 15

    combo_breeds.plot(kind="barh", ax=ax1, color=colors)
    ax1.invert_yaxis()
    ax1.set_title("Top 15 (Blue) vs Bottom 15 (Orange) Breeds by Image Count", fontsize=12)
    ax1.set_xlabel("Number of Images")
    ax1.grid(axis="x", linestyle="--", alpha=0.6)

    # Distribution histogram of counts per breed
    ax2.hist(breed_counts.values, bins=20, color="#2ca02c", edgecolor="black", alpha=0.75)
    ax2.axvline(mean_count, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_count:.1f}")
    ax2.axvline(median_count, color="blue", linestyle=":", linewidth=2, label=f"Median: {median_count:.1f}")
    ax2.set_title("Distribution of Images per Breed Across 120 Classes", fontsize=12)
    ax2.set_xlabel("Images per Breed")
    ax2.set_ylabel("Number of Breeds")
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.6)

    plt.suptitle(f"Dog Breed Class Distribution (Total Classes: {num_breeds}, Total Images: {total_training_images})", fontsize=15)
    plt.tight_layout()
    class_plot_path = output_dir / "class_distribution.png"
    plt.savefig(class_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved class distribution plot to: {class_plot_path}")

    # 3. Dimension Distribution Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Width histogram
    axs[0].hist(valid_dims["width"], bins=30, color="#4c72b0", edgecolor="black", alpha=0.75)
    axs[0].axvline(mean_width, color="red", linestyle="--", label=f"Mean: {mean_width:.0f}px")
    axs[0].axvline(median_width, color="black", linestyle=":", label=f"Median: {median_width:.0f}px")
    axs[0].set_title("Image Width Distribution")
    axs[0].set_xlabel("Width (px)")
    axs[0].set_ylabel("Count")
    axs[0].legend()
    axs[0].grid(linestyle="--", alpha=0.5)

    # Height histogram
    axs[1].hist(valid_dims["height"], bins=30, color="#55a868", edgecolor="black", alpha=0.75)
    axs[1].axvline(mean_height, color="red", linestyle="--", label=f"Mean: {mean_height:.0f}px")
    axs[1].axvline(median_height, color="black", linestyle=":", label=f"Median: {median_height:.0f}px")
    axs[1].set_title("Image Height Distribution")
    axs[1].set_xlabel("Height (px)")
    axs[1].set_ylabel("Count")
    axs[1].legend()
    axs[1].grid(linestyle="--", alpha=0.5)

    # Aspect ratio
    aspect_ratios = valid_dims["width"] / valid_dims["height"]
    axs[2].hist(aspect_ratios, bins=30, color="#c44e52", edgecolor="black", alpha=0.75)
    axs[2].axvline(1.0, color="black", linestyle="--", label="Square (1:1)")
    axs[2].set_title("Aspect Ratio Distribution (Width / Height)")
    axs[2].set_xlabel("Aspect Ratio")
    axs[2].set_ylabel("Count")
    axs[2].legend()
    axs[2].grid(linestyle="--", alpha=0.5)

    # Width vs Height scatter sample (1000 points)
    sample_scatter = valid_dims.sample(n=min(1000, len(valid_dims)), random_state=42)
    axs[3].scatter(sample_scatter["width"], sample_scatter["height"], alpha=0.4, color="#8172b3", s=20)
    axs[3].set_title("Width vs Height Scatter (Sample of 1,000 images)")
    axs[3].set_xlabel("Width (px)")
    axs[3].set_ylabel("Height (px)")
    axs[3].grid(linestyle="--", alpha=0.5)

    plt.suptitle("Image Dimension Characteristics Analysis", fontsize=15)
    plt.tight_layout()
    dim_plot_path = output_dir / "dimension_distribution.png"
    plt.savefig(dim_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved dimension distribution plot to: {dim_plot_path}")

    # Sample verified mappings for demonstration
    sample_mappings = []
    for _, row in df_matched.head(8).iterrows():
        sample_mappings.append({
            "id": row["id"],
            "filename": row["filename"],
            "breed": row["breed"]
        })

    # Summary dictionary
    summary = {
        "dataset_path": str(data_dir),
        "labels_csv_path": str(labels_csv_path),
        "total_files_in_image_directory": total_images,
        "total_training_images": total_training_images,
        "total_breeds": num_breeds,
        "file_extensions": dict(extension_counter),
        "color_modes": color_mode_counts,
        "image_formats": format_counts,
        "missing_images_count": len(labels_without_images),
        "orphaned_images_count": len(images_without_labels),
        "duplicate_ids_in_csv": int(duplicate_ids_count),
        "corrupted_images_count": int(corrupted_count),
        "unreadable_images_count": int(unreadable_count),
        "extremely_small_images_count": int(extremely_small_count),
        "dimensions": {
            "min_width": min_width,
            "max_width": max_width,
            "mean_width": round(mean_width, 2),
            "median_width": round(median_width, 2),
            "min_height": min_height,
            "max_height": max_height,
            "mean_height": round(mean_height, 2),
            "median_height": round(median_height, 2),
            "most_common_dimension": list(most_common_dim),
            "most_common_dimension_frequency": int(most_common_dim_freq),
            "rgb_images": rgb_count,
            "grayscale_images": grayscale_count
        },
        "class_distribution": {
            "most_common_breed": most_common_breed,
            "most_common_count": most_common_count,
            "least_common_breed": least_common_breed,
            "least_common_count": least_common_count,
            "average_images_per_breed": round(mean_count, 2),
            "median_images_per_breed": round(median_count, 2),
            "std_images_per_breed": round(std_count, 2),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "balance_verdict": balance_verdict
        },
        "sample_mappings": sample_mappings
    }

    # Save JSON summary
    summary_json_path = output_dir / "dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved dataset summary to: {summary_json_path}")

    # Generate text report
    report_text_path = output_dir / "dataset_analysis_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1 — DATASET VERIFICATION & STATISTICAL REPORT\n")
        f.write("Dog Breed Classification Using MobileNetV2-Based Transfer Learning\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET PATHS & STRUCTURE\n")
        f.write(f"   Image Directory:       {data_dir}\n")
        f.write(f"   Labels CSV File:       {labels_csv_path}\n")
        f.write(f"   Total Images in Dir:   {total_images}\n")
        f.write(f"   Total Labeled Images:  {total_training_images}\n")
        f.write(f"   File Extensions:       {dict(extension_counter)}\n\n")

        f.write("2. LABEL-TO-IMAGE MAPPING & INTEGRITY\n")
        f.write(f"   Duplicate IDs in CSV:  {duplicate_ids_count}\n")
        f.write(f"   Labels Without Images: {len(labels_without_images)}\n")
        f.write(f"   Images Without Labels: {len(images_without_labels)}\n")
        f.write(f"   Corrupted Images:      {corrupted_count}\n")
        f.write(f"   Unreadable Images:     {unreadable_count}\n")
        f.write(f"   Extremely Small (<32): {extremely_small_count}\n\n")

        f.write("3. SAMPLE VERIFIED MAPPINGS (id -> filename -> breed)\n")
        for sm in sample_mappings:
            f.write(f"   {sm['filename']} -> {sm['breed']} (ID: {sm['id']})\n")
        f.write("\n")

        f.write("4. CLASS STATISTICS (BREEDS)\n")
        f.write(f"   Total Unique Breeds:   {num_breeds}\n")
        f.write(f"   Most Represented:      {most_common_breed} ({most_common_count} images)\n")
        f.write(f"   Least Represented:     {least_common_breed} ({least_common_count} images)\n")
        f.write(f"   Average Images/Breed:  {mean_count:.2f}\n")
        f.write(f"   Median Images/Breed:   {median_count:.2f}\n")
        f.write(f"   Standard Deviation:    {std_count:.2f}\n")
        f.write(f"   Imbalance Ratio:       {imbalance_ratio:.2f}:1\n")
        f.write(f"   Balance Assessment:    {balance_verdict}\n\n")

        f.write("5. IMAGE DIMENSION STATISTICS\n")
        f.write(f"   Width Range:           {min_width}px to {max_width}px (Mean: {mean_width:.1f}px, Median: {median_width:.1f}px)\n")
        f.write(f"   Height Range:          {min_height}px to {max_height}px (Mean: {mean_height:.1f}px, Median: {median_height:.1f}px)\n")
        f.write(f"   Most Common Dimension: {most_common_dim[0]}x{most_common_dim[1]} (Count: {most_common_dim_freq})\n")
        f.write(f"   RGB Images:            {rgb_count}\n")
        f.write(f"   Grayscale Images:      {grayscale_count}\n\n")

        f.write("6. GENERATED ARTIFACTS IN outputs/\n")
        f.write("   - sample_training_images.png (16-sample labeled preview grid)\n")
        f.write("   - class_distribution.png     (Top/bottom classes & distribution histogram)\n")
        f.write("   - dimension_distribution.png (Width/height/aspect-ratio analysis)\n")
        f.write("   - dataset_summary.json       (Complete machine-readable metrics)\n")
        f.write("   - breed_distribution.csv     (Per-breed counts and proportions)\n")
        f.write("   - breed_names.txt            (Alphabetical index of all 120 breeds)\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Saved written report to: {report_text_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Dataset Verification and Analysis")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing training images")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output charts and reports")
    parser.add_argument("--sample-size", type=int, default=16, help="Number of sample images to plot")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for image checking")

    args = parser.parse_args()

    # Determine default output directory relative to script or workspace
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Check if outputs/ directory exists in current working dir or in script parent
        script_dir = Path(__file__).resolve().parent
        proj_root = script_dir.parent
        if (proj_root / "outputs").is_dir():
            output_dir = proj_root / "outputs"
        elif Path("outputs").is_dir():
            output_dir = Path("outputs")
        elif Path("dog_breed_classification/outputs").is_dir():
            output_dir = Path("dog_breed_classification/outputs")
        else:
            output_dir = proj_root / "outputs"

    check_environment()
    data_dir, labels_csv = find_dataset(args.data_dir, args.labels)
    logger.info(f"Using Image Directory: {data_dir.resolve()}")
    logger.info(f"Using Labels CSV:      {labels_csv.resolve()}")
    logger.info(f"Outputs will be saved: {output_dir.resolve()}")

    summary = analyze_dataset(
        data_dir=data_dir,
        labels_csv_path=labels_csv,
        output_dir=output_dir,
        sample_size=args.sample_size,
        num_workers=args.workers
    )

    logger.info("Phase 1 Dataset Verification completed successfully.")
    return summary


if __name__ == "__main__":
    main()
