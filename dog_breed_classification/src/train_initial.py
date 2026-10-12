"""
Initial MobileNetV2 Training Script (Frozen Feature Extractor Stage)
Phase 4 - Dog Breed Classification Using MobileNetV2-Based Transfer Learning

This module performs:
1. Loading verified datasets (train and validation) from Phase 2
2. Building and verifying the MobileNetV2 transfer learning model
3. Strict verification of base model freezing (base_model.trainable = False)
4. Configuring callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
5. Training the classification head for up to 30 epochs
6. Saving the best model to models/mobilenetv2_initial_best.keras
7. Recording metrics history to outputs/training_history.csv
8. Generating separate training graphs:
   - outputs/initial_accuracy.png
   - outputs/initial_loss.png
9. Analyzing overfitting and loss divergence
10. Validating the saved best model via an inference reload test
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

# Import modular project components from Phase 2 and 3
try:
    from data_preprocessing import (
        create_dataframe,
        split_dataset,
        create_dataset,
        resolve_paths as resolve_data_paths
    )
    from model import (
        build_model,
        verify_trainable_parameters,
        get_num_classes,
        resolve_paths as resolve_model_paths
    )
except ModuleNotFoundError:
    from dog_breed_classification.src.data_preprocessing import (
        create_dataframe,
        split_dataset,
        create_dataset,
        resolve_paths as resolve_data_paths
    )
    from dog_breed_classification.src.model import (
        build_model,
        verify_trainable_parameters,
        get_num_classes,
        resolve_paths as resolve_model_paths
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Training configuration constants
DEFAULT_MAX_EPOCHS = 13
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_REDUCE_LR_PATIENCE = 2
DEFAULT_REDUCE_LR_FACTOR = 0.2
DEFAULT_MIN_LR = 1e-6


def resolve_project_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Resolves project root, models directory, outputs directory, and dataset directory.
    Uses relative path discovery without hardcoded absolute paths.
    """
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent

    models_dir = proj_root / "models"
    outputs_dir = proj_root / "outputs"
    dataset_dir = proj_root / "dataset"

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return proj_root, models_dir, outputs_dir, dataset_dir


def load_training_data(
    outputs_dir: Path,
    dataset_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> Tuple[tf.data.Dataset, tf.data.Dataset, pd.DataFrame, pd.DataFrame, int]:
    """
    Loads train and validation splits.
    Prefers the exact split CSVs saved in Phase 2 to ensure zero divergence.
    """
    train_csv_path = outputs_dir / "train_split.csv"
    val_csv_path = outputs_dir / "val_split.csv"
    class_names_path = outputs_dir / "class_names.json"

    if train_csv_path.is_file() and val_csv_path.is_file() and class_names_path.is_file():
        logger.info(f"Loading preserved split CSVs from {outputs_dir}...")
        train_df = pd.read_csv(train_csv_path)
        val_df = pd.read_csv(val_csv_path)

        with open(class_names_path, "r", encoding="utf-8") as f:
            class_meta = json.load(f)
        num_classes = class_meta["num_classes"]
    else:
        logger.info("Preserved split CSVs not found in outputs/. Regenerating from dataset...")
        data_dir, labels_csv, _ = resolve_data_paths()
        df, class_to_id, _, class_names = create_dataframe(data_dir, labels_csv)
        num_classes = len(class_names)
        train_df, val_df, _ = split_dataset(df)

    logger.info(f"Training samples:   {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")
    logger.info(f"Number of classes:  {num_classes}")

    # Build tf.data pipelines
    # Note: data augmentation is integrated directly into the model architecture
    train_ds = create_dataset(
        df=train_df,
        batch_size=batch_size,
        is_training=True,
        augment_model=None
    )

    val_ds = create_dataset(
        df=val_df,
        batch_size=batch_size,
        is_training=False,
        augment_model=None
    )

    return train_ds, val_ds, train_df, val_df, num_classes


def load_class_weights(outputs_dir: Path) -> Optional[Dict[int, float]]:
    """Loads balanced class weights computed during Phase 2."""
    weights_path = outputs_dir / "class_weights.json"
    if weights_path.is_file():
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_weights = data.get("class_weights", {})
            class_weights = {int(k): float(v) for k, v in raw_weights.items()}
            logger.info(f"Loaded balanced class weights for {len(class_weights)} classes.")
            return class_weights
        except Exception as e:
            logger.warning(f"Could not load class weights from {weights_path}: {e}")
    return None


def verify_frozen_model(model: tf.keras.Model, base_model: tf.keras.Model) -> Dict[str, Any]:
    """
    Verifies that the MobileNetV2 base model is strictly frozen and only the
    classification head contains trainable parameters.
    """
    assert base_model.trainable is False, "CRITICAL ERROR: base_model.trainable is True! Base must be frozen."

    param_info = verify_trainable_parameters(model)

    # Detailed assertion checks
    assert param_info["non_trainable_params"] == 2257984, (
        f"Unexpected non-trainable parameter count: {param_info['non_trainable_params']} (expected 2,257,984)"
    )
    assert param_info["trainable_params"] > 0, "No trainable parameters in classification head!"

    logger.info("Verification PASSED: MobileNetV2 feature extractor is 100% FROZEN.")
    logger.info(f"Classification head is TRAINABLE with {param_info['trainable_params']:,} parameters.")
    return param_info


def setup_callbacks(
    model_save_path: Path,
    history_csv_path: Path,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = DEFAULT_REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = DEFAULT_REDUCE_LR_FACTOR,
    min_lr: float = DEFAULT_MIN_LR
) -> list:
    """Configures Keras callbacks for training monitoring and model checkpointing."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(history_csv_path),
            separator=",",
            append=False
        )
    ]
    return callbacks


def plot_accuracy_graph(history_df: pd.DataFrame, output_path: Path, best_epoch: int) -> None:
    """Generates standalone Training vs Validation Accuracy graph."""
    plt.figure(figsize=(10, 6), dpi=200)

    epochs = history_df["epoch"]
    train_acc = history_df["accuracy"]
    val_acc = history_df["val_accuracy"]

    plt.plot(epochs, train_acc, "o-", color="#1f77b4", linewidth=2.2, label="Training Accuracy")
    plt.plot(epochs, val_acc, "s-", color="#ff7f0e", linewidth=2.2, label="Validation Accuracy")

    # Highlight best epoch
    best_val_acc = val_acc.iloc[best_epoch - 1]
    plt.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    plt.scatter([best_epoch], [best_val_acc], color="#2ca02c", s=120, zorder=5)

    plt.title("Initial MobileNetV2 Training — Accuracy Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy graph to: {output_path}")


def plot_loss_graph(history_df: pd.DataFrame, output_path: Path, best_epoch: int) -> None:
    """Generates standalone Training vs Validation Loss graph."""
    plt.figure(figsize=(10, 6), dpi=200)

    epochs = history_df["epoch"]
    train_loss = history_df["loss"]
    val_loss = history_df["val_loss"]

    plt.plot(epochs, train_loss, "o-", color="#d62728", linewidth=2.2, label="Training Loss")
    plt.plot(epochs, val_loss, "s-", color="#9467bd", linewidth=2.2, label="Validation Loss")

    # Highlight best epoch
    best_val_loss = val_loss.iloc[best_epoch - 1]
    plt.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch})")
    plt.scatter([best_epoch], [best_val_loss], color="#2ca02c", s=120, zorder=5)

    plt.title("Initial MobileNetV2 Training — Loss Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Sparse Categorical Cross-Entropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss graph to: {output_path}")


def analyze_overfitting(history_df: pd.DataFrame, best_epoch: int) -> Dict[str, Any]:
    """Analyzes training curves for signs of overfitting or divergence."""
    best_row = history_df[history_df["epoch"] == best_epoch].iloc[0]
    final_row = history_df.iloc[-1]

    train_acc_final = float(final_row["accuracy"])
    val_acc_final = float(final_row["val_accuracy"])
    acc_gap = train_acc_final - val_acc_final

    train_loss_final = float(final_row["loss"])
    val_loss_final = float(final_row["val_loss"])
    loss_gap = val_loss_final - train_loss_final

    # Diagnostic logic
    is_overfitting = bool(acc_gap > 0.20 or (val_loss_final > best_row["val_loss"] * 1.15))
    if is_overfitting:
        diagnosis = (
            f"Mild overfitting observed towards later epochs: Training accuracy ({train_acc_final:.3f}) "
            f"exceeds validation accuracy ({val_acc_final:.3f}) by {acc_gap:.3f}. "
            f"EarlyStopping successfully restored best weights from Epoch {best_epoch}."
        )
    else:
        diagnosis = (
            f"Model shows healthy generalization during feature extraction stage: "
            f"Validation loss closely tracks training loss (gap: {loss_gap:.3f}). "
            f"Dropout(0.3) and Data Augmentation effectively regularized the classification head."
        )

    logger.info(f"Overfitting Analysis: {diagnosis}")
    return {
        "is_overfitting": is_overfitting,
        "acc_gap": round(acc_gap, 4),
        "loss_gap": round(loss_gap, 4),
        "diagnosis": diagnosis
    }


def verify_saved_model(
    model_save_path: Path,
    num_classes: int,
    batch_size: int = 4
) -> Dict[str, Any]:
    """
    Reloads the saved .keras model and runs a miniature batch forward pass
    to verify model integrity, probability outputs, and shapes.
    """
    logger.info(f"Verifying saved model reload from {model_save_path}...")
    assert model_save_path.is_file(), f"Saved model file does not exist at {model_save_path}!"

    loaded_model = tf.keras.models.load_model(str(model_save_path))

    dummy_input = tf.random.uniform((batch_size, 224, 224, 3), minval=-1.0, maxval=1.0)
    preds = loaded_model(dummy_input, training=False)
    preds_np = preds.numpy()

    output_shape = list(preds_np.shape)
    sums_per_sample = np.sum(preds_np, axis=1)
    has_nan = bool(np.isnan(preds_np).any())
    has_inf = bool(np.isinf(preds_np).any())

    assert output_shape == [batch_size, num_classes], f"Reloaded model shape mismatch: {output_shape}"
    assert not has_nan, "NaN values detected in reloaded model output!"
    assert not has_inf, "Inf values detected in reloaded model output!"
    assert np.allclose(sums_per_sample, 1.0, atol=1e-4), "Softmax probabilities do not sum to 1.0!"

    logger.info(f"Reload verification PASSED: Output shape {output_shape}, Softmax sums {sums_per_sample.round(4)}")
    return {
        "reload_verified": True,
        "output_shape": output_shape,
        "sums_approx_one": True,
        "has_nan": has_nan,
        "has_inf": has_inf
    }


def train_initial_model(
    epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE
) -> Dict[str, Any]:
    """
    Executes Phase 4 initial training of MobileNetV2 classification head.
    """
    proj_root, models_dir, outputs_dir, dataset_dir = resolve_project_paths()
    model_save_path = models_dir / "mobilenetv2_initial_best.keras"
    history_csv_path = outputs_dir / "training_history.csv"
    acc_plot_path = outputs_dir / "initial_accuracy.png"
    loss_plot_path = outputs_dir / "initial_loss.png"

    logger.info("=== PHASE 4: INITIAL MOBILENETV2 TRAINING ===")
    logger.info(f"Target Epochs:     {epochs}")
    logger.info(f"Batch Size:        {batch_size}")
    logger.info(f"Learning Rate:     {learning_rate}")
    logger.info(f"Model Save Path:   {model_save_path}")
    logger.info(f"History CSV Path:  {history_csv_path}")

    # 1. Load Data
    train_ds, val_ds, train_df, val_df, num_classes = load_training_data(
        outputs_dir=outputs_dir,
        dataset_dir=dataset_dir,
        batch_size=batch_size
    )

    # 2. Build Model
    model, base_model = build_model(
        num_classes=num_classes,
        include_augmentation=True,
        learning_rate=learning_rate
    )

    # 3. Verify Freezing
    param_info = verify_frozen_model(model, base_model)

    # 4. Setup Callbacks
    callbacks = setup_callbacks(
        model_save_path=model_save_path,
        history_csv_path=history_csv_path,
        early_stopping_patience=DEFAULT_EARLY_STOPPING_PATIENCE,
        reduce_lr_patience=DEFAULT_REDUCE_LR_PATIENCE
    )

    # 5. Class weights
    class_weights = load_class_weights(outputs_dir)

    # 6. Fit Model
    logger.info("Starting model training (classification head only)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # 7. Process Training History
    # Load history from CSVLogger (guarantees exact written metrics)
    if history_csv_path.is_file():
        history_df = pd.read_csv(history_csv_path)
    else:
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = range(1, len(history_df) + 1)
        history_df.to_csv(history_csv_path, index=False)

    # Normalize epoch column (1-indexed)
    if "epoch" in history_df.columns:
        if history_df["epoch"].min() == 0:
            history_df["epoch"] = history_df["epoch"] + 1
    else:
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    # Add learning rate if missing from CSV
    if "learning_rate" not in history_df.columns and "lr" in history.history:
        history_df["learning_rate"] = history.history["lr"]

    epochs_completed = len(history_df)
    # Best epoch based on minimum val_loss
    best_idx = history_df["val_loss"].idxmin()
    best_epoch = int(history_df.loc[best_idx, "epoch"])
    best_val_loss = float(history_df.loc[best_idx, "val_loss"])
    best_val_acc = float(history_df.loc[best_idx, "val_accuracy"])
    best_train_loss = float(history_df.loc[best_idx, "loss"])
    best_train_acc = float(history_df.loc[best_idx, "accuracy"])

    # 8. Generate Separate Visualizations
    plot_accuracy_graph(history_df, acc_plot_path, best_epoch)
    plot_loss_graph(history_df, loss_plot_path, best_epoch)

    # 9. Overfitting Analysis
    overfitting_info = analyze_overfitting(history_df, best_epoch)

    # 10. Verify Model Reload
    reload_verification = verify_saved_model(model_save_path, num_classes)

    # 11. Compile Phase 4 Results
    phase_4_summary = {
        "epochs_completed": epochs_completed,
        "max_epochs_configured": epochs,
        "best_epoch": best_epoch,
        "best_training_accuracy": round(best_train_acc, 4),
        "best_validation_accuracy": round(best_val_acc, 4),
        "best_training_loss": round(best_train_loss, 4),
        "best_validation_loss": round(best_val_loss, 4),
        "trainable_parameters": param_info["trainable_params"],
        "non_trainable_parameters": param_info["non_trainable_params"],
        "saved_model_path": str(model_save_path),
        "training_history_path": str(history_csv_path),
        "accuracy_graph_path": str(acc_plot_path),
        "loss_graph_path": str(loss_plot_path),
        "overfitting_analysis": overfitting_info,
        "reload_verification": reload_verification,
        "status": "INITIAL TRAINING COMPLETE — READY FOR PHASE 5 FINE-TUNING"
    }

    # Save summary JSON
    summary_json_path = outputs_dir / "initial_training_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(phase_4_summary, f, indent=2)
    logger.info(f"Saved initial training summary to: {summary_json_path}")

    # 12. Print Formatted Summary
    print("\n" + "=" * 50)
    print("INITIAL TRAINING RESULTS")
    print("=" * 50)
    print(f"Epochs completed:         {epochs_completed} / {epochs}")
    print(f"Best epoch:               {best_epoch}")
    print(f"Best training accuracy:   {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best training loss:       {best_train_loss:.4f}")
    print(f"Best validation loss:     {best_val_loss:.4f}")
    print(f"Trainable parameters:     {param_info['trainable_params']:,}")
    print(f"Frozen parameters:        {param_info['non_trainable_params']:,}")
    print(f"Saved model path:         {model_save_path}")
    print(f"Training history path:    {history_csv_path}")
    print(f"Accuracy graph path:      {acc_plot_path}")
    print(f"Loss graph path:          {loss_plot_path}")
    print(f"Overfitting diagnosis:    {overfitting_info['diagnosis']}")
    print("=" * 50 + "\n")

    return phase_4_summary


if __name__ == "__main__":
    train_initial_model()
