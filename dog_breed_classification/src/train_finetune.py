
"""
Phase 5 - MobileNetV2 Fine-Tuning Script
Dog Breed Classification Using MobileNetV2-Based Transfer Learning
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

try:
    from data_preprocessing import create_dataframe, split_dataset, create_dataset, resolve_paths as resolve_data_paths
    from model import build_model, verify_trainable_parameters, get_num_classes, resolve_paths as resolve_model_paths
except ModuleNotFoundError:
    from dog_breed_classification.src.data_preprocessing import create_dataframe, split_dataset, create_dataset, resolve_paths as resolve_data_paths
    from dog_breed_classification.src.model import build_model, verify_trainable_parameters, get_num_classes, resolve_paths as resolve_model_paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

DEFAULT_MAX_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_REDUCE_LR_PATIENCE = 2
DEFAULT_REDUCE_LR_FACTOR = 0.2
DEFAULT_MIN_LR = 1e-7
FROZEN_LAYER_COUNT = 80
# Set to >0 to resume training from a specific epoch (load finetuned checkpoint).
RESUME_FROM_EPOCH = 4


def resolve_project_paths():
    script_dir = Path(__file__).resolve().parent
    proj_root = script_dir.parent
    models_dir = proj_root / "models"
    outputs_dir = proj_root / "outputs"
    dataset_dir = proj_root / "dataset"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return proj_root, models_dir, outputs_dir, dataset_dir


def load_phase4_model(models_dir, resume_from_epoch=0):
    """Load the finetuned checkpoint for resume, otherwise the Phase 4 model."""
    finetuned_path = models_dir / "mobilenetv2_finetuned_best.keras"
    if resume_from_epoch > 0 and finetuned_path.is_file():
        logger.info("RESUME MODE: Loading finetuned checkpoint from " + str(finetuned_path))
        return tf.keras.models.load_model(str(finetuned_path))
    model_path = models_dir / "mobilenetv2_initial_best.keras"
    assert model_path.is_file(), "Phase 4 model not found at " + str(model_path)
    logger.info("Loading Phase 4 model from " + str(model_path))
    return tf.keras.models.load_model(str(model_path))


def unfreeze_mobile_net(model, frozen_count):
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not locate MobileNetV2 base inside the loaded model.")
    total_layers = len(base_model.layers)
    logger.info(f"MobileNetV2 has {total_layers} total layers.")
    logger.info(f"Keeping first {frozen_count} layers frozen; unfreezing the rest (BN always frozen).")
    for i, layer in enumerate(base_model.layers):
        if i < frozen_count:
            layer.trainable = False
        else:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
    frozen = sum(1 for l in base_model.layers if not l.trainable)
    trainable = sum(1 for l in base_model.layers if l.trainable)
    return frozen, trainable


def recompile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")]
    )
    logger.info(f"Model recompiled. LR={lr}")


def setup_callbacks(model_save_path, history_csv_path, append_csv=False):
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=DEFAULT_EARLY_STOPPING_PATIENCE, restore_best_weights=True, mode="min", verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_save_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=DEFAULT_REDUCE_LR_FACTOR, patience=DEFAULT_REDUCE_LR_PATIENCE, min_lr=DEFAULT_MIN_LR, mode="min", verbose=1),
        tf.keras.callbacks.CSVLogger(filename=str(history_csv_path), separator=",", append=append_csv),
    ]


def load_training_data(outputs_dir, dataset_dir, batch_size):
    train_csv = outputs_dir / "train_split.csv"
    val_csv = outputs_dir / "val_split.csv"
    class_names_path = outputs_dir / "class_names.json"
    if not (train_csv.is_file() and val_csv.is_file() and class_names_path.is_file()):
        raise FileNotFoundError("Split CSVs or class_names.json not found in outputs/. Run Phase 2 first.")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    with open(class_names_path) as f:
        class_meta = json.load(f)
    num_classes = class_meta["num_classes"]
    train_ds = create_dataset(df=train_df, batch_size=batch_size, is_training=True, augment_model=None)
    val_ds = create_dataset(df=val_df, batch_size=batch_size, is_training=False, augment_model=None)
    logger.info("Train/val splits loaded for fine-tuning.")
    return train_ds, val_ds, train_df, val_df, num_classes


def load_class_weights(outputs_dir):
    weights_path = outputs_dir / "class_weights.json"
    if weights_path.is_file():
        try:
            with open(weights_path) as f:
                data = json.load(f)
            raw = data.get("class_weights", {})
            return {int(k): float(v) for k, v in raw.items()}
        except Exception as e:
            logger.warning(f"Could not load class weights: {e}")
    return None


def plot_graph(history_df, metric, val_metric, output_path, best_epoch, title, y_label):
    plt.figure(figsize=(10, 6), dpi=200)
    epochs = history_df["epoch"]
    plt.plot(epochs, history_df[metric], "o-", color="#1f77b4", linewidth=2.2, label="Training " + metric)
    plt.plot(epochs, history_df[val_metric], "s-", color="#ff7f0e", linewidth=2.2, label="Validation " + metric)
    best_val = history_df[val_metric].iloc[best_epoch - 1]
    plt.axvline(best_epoch, color="#2ca02c", linestyle="--", alpha=0.8, label="Best Epoch (" + str(best_epoch) + ")")
    plt.scatter([best_epoch], [best_val], color="#2ca02c", s=120, zorder=5)
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved graph to " + str(output_path))


def compare_phases(p4_path, p5_path, out_path):
    with open(p4_path) as f: p4 = json.load(f)
    with open(p5_path) as f: p5 = json.load(f)
    acc_delta = p5["best_validation_accuracy"] - p4["best_validation_accuracy"]
    loss_delta = p4["best_validation_loss"] - p5["best_validation_loss"]
    p4_top5 = p4.get("best_validation_top_5_accuracy", 0)
    p5_top5 = p5.get("best_validation_top_5_accuracy", 0)
    improved = p5["best_validation_accuracy"] >= p4["best_validation_accuracy"]
    selected = "mobilenetv2_finetuned_best.keras" if improved else "mobilenetv2_initial_best.keras"
    lines = [
        "=" * 60,
        "Phase 4 vs Phase 5 Comparison",
        "=" * 60,
        "",
        "Phase 4 best val accuracy : " + str(round(p4["best_validation_accuracy"], 4)),
        "Phase 5 best val accuracy : " + str(round(p5["best_validation_accuracy"], 4)),
        "Accuracy change           : " + ("+{:.4f}".format(acc_delta) if acc_delta >= 0 else "{:.4f}".format(acc_delta)),
        "",
        "Phase 4 best val loss     : " + str(round(p4["best_validation_loss"], 4)),
        "Phase 5 best val loss     : " + str(round(p5["best_validation_loss"], 4)),
        "Loss change               : " + ("+{:.4f}".format(loss_delta) if loss_delta >= 0 else "{:.4f}".format(loss_delta)) + " (positive = improvement)",
        "",
        "Phase 4 best val top-5    : " + str(round(p4_top5, 4)),
        "Phase 5 best val top-5    : " + str(round(p5_top5, 4)),
        "",
        "Fine-tuning improved model: " + ("YES" if improved else "NO"),
        "Selected model for Phase 6: " + selected,
        "",
        "Overfitting assessment: Review finetuning_accuracy.png and finetuning_loss.png.",
        "Significant divergence between training and validation curves indicates overfitting.",
        "=" * 60,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison saved to " + str(out_path))
    return improved, selected, acc_delta, loss_delta


def main():
    proj_root, models_dir, outputs_dir, dataset_dir = resolve_project_paths()

    train_ds, val_ds, train_df, val_df, num_classes = load_training_data(outputs_dir, dataset_dir, DEFAULT_BATCH_SIZE)

    # Resume from checkpoint if RESUME_FROM_EPOCH > 0
    resuming = RESUME_FROM_EPOCH > 0
    model = load_phase4_model(models_dir, resume_from_epoch=RESUME_FROM_EPOCH)
    logger.info("Model total parameters: " + str(model.count_params()))

    dummy = tf.zeros((1, 224, 224, 3))
    out = model(dummy, training=False)
    assert out.shape[-1] == 120, "Expected 120 output classes, got " + str(out.shape[-1])
    logger.info("Architecture verified: 120 output classes confirmed.")

    frozen, trainable = unfreeze_mobile_net(model, FROZEN_LAYER_COUNT)
    recompile_model(model, DEFAULT_LEARNING_RATE)
    param_info = verify_trainable_parameters(model)

    print("\n" + "=" * 50)
    print("LAYER CONFIGURATION BEFORE TRAINING")
    print("=" * 50)
    print("Total parameters       : " + str(model.count_params()))
    print("Trainable parameters   : " + str(param_info["trainable_params"]))
    print("Non-trainable params   : " + str(param_info["non_trainable_params"]))
    print("Frozen MobileNet layers: " + str(frozen))
    print("Trainable MobileNet layers: " + str(trainable))
    if resuming:
        print("Resuming from epoch    : " + str(RESUME_FROM_EPOCH + 1))
    print("=" * 50 + "\n")

    finetuned_path = models_dir / "mobilenetv2_finetuned_best.keras"
    history_csv = outputs_dir / "finetuning_history.csv"
    # Append to existing CSV when resuming so full history is preserved
    callbacks = setup_callbacks(finetuned_path, history_csv, append_csv=resuming)
    class_weights = load_class_weights(outputs_dir)

    if resuming:
        logger.info("=== PHASE 5: RESUMING FINE-TUNING FROM EPOCH " + str(RESUME_FROM_EPOCH + 1) + " ===")
    else:
        logger.info("=== PHASE 5: FINE-TUNING START ===")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=RESUME_FROM_EPOCH,
        epochs=DEFAULT_MAX_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    if history_csv.is_file():
        history_df = pd.read_csv(history_csv)
    else:
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = range(1, len(history_df) + 1)
        history_df.to_csv(history_csv, index=False)

    if "epoch" in history_df.columns and history_df["epoch"].min() == 0:
        history_df["epoch"] += 1
    if "learning_rate" not in history_df.columns and "lr" in history.history:
        history_df["learning_rate"] = history.history["lr"]

    epochs_completed = len(history_df)
    best_idx = history_df["val_loss"].idxmin()
    best_epoch = int(history_df.loc[best_idx, "epoch"])
    bv = history_df.loc[best_idx]

    plot_graph(history_df, "accuracy", "val_accuracy", outputs_dir / "finetuning_accuracy.png", best_epoch, "Fine-Tuning - Accuracy Curves", "Accuracy")
    plot_graph(history_df, "loss", "val_loss", outputs_dir / "finetuning_loss.png", best_epoch, "Fine-Tuning - Loss Curves", "Sparse Categorical Cross-Entropy")

    param_info = verify_trainable_parameters(model)
    phase5_summary = {
        "epochs_completed": epochs_completed,
        "max_epochs_configured": DEFAULT_MAX_EPOCHS,
        "best_epoch": best_epoch,
        "best_training_accuracy": round(float(bv["accuracy"]), 4),
        "best_validation_accuracy": round(float(bv["val_accuracy"]), 4),
        "best_training_loss": round(float(bv["loss"]), 4),
        "best_validation_loss": round(float(bv["val_loss"]), 4),
        "best_validation_top_5_accuracy": round(float(bv["val_top_5_accuracy"]), 4),
        "trainable_parameters": param_info["trainable_params"],
        "non_trainable_parameters": param_info["non_trainable_params"],
        "frozen_mobilenet_layers": frozen,
        "trainable_mobilenet_layers": trainable,
        "saved_model_path": str(finetuned_path),
        "status": "FINETUNING COMPLETE",
    }
    summary_json = outputs_dir / "finetuning_summary.json"
    with open(summary_json, "w") as f:
        json.dump(phase5_summary, f, indent=2)
    logger.info("Summary saved to " + str(summary_json))

    improved, selected, acc_delta, loss_delta = compare_phases(
        outputs_dir / "initial_training_summary.json",
        summary_json,
        outputs_dir / "phase4_vs_phase5_comparison.txt"
    )

    logger.info("Reloading finetuned model for verification...")
    loaded = tf.keras.models.load_model(str(finetuned_path))
    logger.info("Reload verification PASSED.")

    with open(outputs_dir / "class_names.json") as f:
        meta = json.load(f)
    class_names = meta.get("class_names", [])

    sample_images, sample_labels = next(iter(val_ds.take(1)))
    preds = loaded(sample_images, training=False).numpy()
    top5_indices = np.argsort(preds, axis=1)[:, ::-1][:, :5]
    pred_classes = top5_indices[:, 0]
    confidences = preds[np.arange(len(preds)), pred_classes]

    print("\nSample Predictions on Validation Set:")
    for i in range(min(5, len(sample_labels))):
        true = class_names[sample_labels[i].numpy()] if class_names else str(sample_labels[i].numpy())
        pred = class_names[pred_classes[i]] if class_names else str(pred_classes[i])
        top5 = [class_names[j] if class_names else str(j) for j in top5_indices[i]]
        print("  [" + str(i+1) + "] True: " + true)
        print("       Pred: " + pred + " (confidence: " + str(round(confidences[i]*100, 2)) + "%)")
        print("       Top-5: " + str(top5))

    with open(outputs_dir / "initial_training_summary.json") as f:
        p4 = json.load(f)
    final_lr = history_df["learning_rate"].iloc[-1] if "learning_rate" in history_df.columns else DEFAULT_LEARNING_RATE

    print("\n" + "=" * 60)
    print("PHASE 5 FINAL REPORT")
    print("=" * 60)
    print("Phase 4 best val accuracy   : " + str(p4["best_validation_accuracy"]))
    print("Phase 5 best val accuracy   : " + str(phase5_summary["best_validation_accuracy"]))
    print("Phase 4 best val loss       : " + str(p4["best_validation_loss"]))
    print("Phase 5 best val loss       : " + str(phase5_summary["best_validation_loss"]))
    print("Phase 4 best val top-5 acc  : " + str(p4.get("best_validation_top_5_accuracy", "N/A")))
    print("Phase 5 best val top-5 acc  : " + str(phase5_summary["best_validation_top_5_accuracy"]))
    print("Epochs completed            : " + str(epochs_completed) + " / " + str(DEFAULT_MAX_EPOCHS))
    print("Best epoch                  : " + str(best_epoch))
    print("Final learning rate         : " + str(final_lr))
    print("Trainable params            : " + str(param_info["trainable_params"]))
    print("Fine-tuning improved model  : " + ("YES" if improved else "NO"))
    print("Selected model for Phase 6  : " + selected)
    print("=" * 60)
    print("\nPhase 5 complete. Waiting for confirmation before Phase 6.")


if __name__ == "__main__":
    main()
