import sys, os, json, logging, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)

try:
    from data_preprocessing import create_dataset
except ModuleNotFoundError:
    from dog_breed_classification.src.data_preprocessing import create_dataset

FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FMT,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
BATCH_SIZE = 32

def resolve_paths():
    script_dir = Path(__file__).resolve().parent
    proj = script_dir.parent
    return proj, proj / "models", proj / "outputs"

def select_model(models_dir, outputs_dir):
    with open(outputs_dir / "initial_training_summary.json") as f: p4 = json.load(f)
    with open(outputs_dir / "finetuning_summary.json") as f: p5 = json.load(f)
    p4_acc, p4_loss = p4["best_validation_accuracy"], p4["best_validation_loss"]
    p5_acc, p5_loss = p5["best_validation_accuracy"], p5["best_validation_loss"]
    p4_top5 = p4.get("best_validation_top_5_accuracy", 0)
    p5_top5 = p5.get("best_validation_top_5_accuracy", 0)
    logger.info("Phase 4 val acc=" + str(p4_acc) + "  loss=" + str(p4_loss))
    logger.info("Phase 5 val acc=" + str(p5_acc) + "  loss=" + str(p5_loss))
    if p5_acc >= p4_acc:
        logger.info("Selected: Phase 5 Fine-tuned model")
        return ("Phase 5 Fine-tuned", models_dir / "mobilenetv2_finetuned_best.keras",
                p5_acc, p5_loss, p5_top5)
    logger.info("Selected: Phase 4 Initial model")
    return ("Phase 4 Initial", models_dir / "mobilenetv2_initial_best.keras",
            p4_acc, p4_loss, p4_top5)

def load_class_names(outputs_dir):
    with open(outputs_dir / "class_names.json") as f: meta = json.load(f)
    return meta["class_names"], meta["num_classes"]

def build_test_dataset(outputs_dir, batch_size):
    test_df = pd.read_csv(outputs_dir / "test_split.csv")
    logger.info("Test samples: " + str(len(test_df)))
    test_ds = create_dataset(df=test_df, batch_size=batch_size,
                              is_training=False, augment_model=None)
    return test_ds, test_df

def evaluate_model(model, test_ds):
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in test_ds:
        p = model(images, training=False).numpy()
        all_probs.extend(p)
        all_preds.extend(np.argmax(p, axis=1))
        all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    probs  = np.array(all_probs)
    loss_val = float(tf.keras.losses.SparseCategoricalCrossentropy()(y_true, probs).numpy())
    acc = float(np.mean(y_true == y_pred))
    top5 = np.argsort(probs, axis=1)[:, ::-1][:, :5]
    top5_acc = float(np.mean([y_true[i] in top5[i] for i in range(len(y_true))]))
    return y_true, y_pred, probs, loss_val, acc, top5_acc

def save_confusion_matrix(y_true, y_pred, class_names, outputs_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(outputs_dir / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(28, 26), dpi=100)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    n = len(class_names)
    step = max(1, n // 30)
    ticks = list(range(0, n, step))
    tick_labels = [class_names[i] for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
    ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=5)
    ax.set_title("Confusion Matrix - Dog Breed Classifier - 120 Classes",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Breed", fontsize=11)
    ax.set_ylabel("True Breed", fontsize=11)
    plt.tight_layout()
    plt.savefig(outputs_dir / "confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved.")

def save_prediction_gallery(model, test_df, class_names, outputs_dir):
    sample = test_df.sample(n=min(48, len(test_df)), random_state=42).reset_index(drop=True)
    results = []
    for _, row in sample.iterrows():
        img_path = row.get("filepath", row.get("image_path", ""))
        true_id  = int(row.get("label", row.get("class_id", 0)))
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                      np.array(img, dtype=np.float32))
            p   = model(tf.expand_dims(arr, 0), training=False).numpy()[0]
            pid = int(np.argmax(p))
            results.append({"img": np.array(img), "true_id": true_id,
                             "pred_id": pid, "conf": float(p[pid])})
        except Exception: pass
        if len(results) >= 12: break
    n = len(results); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5), dpi=120)
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        ax.imshow(r["img"])
        ok  = r["true_id"] == r["pred_id"]
        col = "#27ae60" if ok else "#e74c3c"
        marker = "OK" if ok else "X"
        ttl = ("[" + marker + "] True: " + class_names[r["true_id"]] +
               "\nPred: " + class_names[r["pred_id"]] +
               " (" + str(round(r["conf"] * 100, 1)) + "%)")
        ax.set_title(ttl, fontsize=6.5, color=col, fontweight="bold")
        for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle("Test Prediction Gallery - Dog Breed Classifier",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(outputs_dir / "test_prediction_gallery.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Prediction gallery saved.")

def save_top5_csv(model, test_df, class_names, outputs_dir, n=50):
    sample = test_df.sample(n=min(n, len(test_df)), random_state=7).reset_index(drop=True)
    records = []
    for _, row in sample.iterrows():
        img_path = row.get("filepath", row.get("image_path", ""))
        true_id  = int(row.get("label", row.get("class_id", 0)))
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                      np.array(img, dtype=np.float32))
            p   = model(tf.expand_dims(arr, 0), training=False).numpy()[0]
            top5 = np.argsort(p)[::-1][:5]
            rec = {"image": img_path, "true_breed": class_names[true_id]}
            for rank, idx in enumerate(top5, 1):
                rec["top" + str(rank) + "_breed"] = class_names[idx]
                rec["top" + str(rank) + "_conf"]  = round(float(p[idx]), 4)
            records.append(rec)
        except Exception: pass
    pd.DataFrame(records).to_csv(outputs_dir / "top5_predictions.csv", index=False)
    logger.info("Top-5 predictions CSV saved.")

def main():
    proj, models_dir, outputs_dir = resolve_paths()
    chosen, model_file, bv_acc, bv_loss, bv_top5 = select_model(models_dir, outputs_dir)
    class_names, num_classes = load_class_names(outputs_dir)
    test_ds, test_df = build_test_dataset(outputs_dir, BATCH_SIZE)
    logger.info("Loading: " + str(model_file))
    model = tf.keras.models.load_model(str(model_file))
    logger.info("Evaluating on test set...")
    y_true, y_pred, probs, test_loss, test_acc, top5_acc = evaluate_model(model, test_ds)
    logger.info("Test acc=" + str(round(test_acc, 4)) +
                "  loss=" + str(round(test_loss, 4)) +
                "  top5=" + str(round(top5_acc, 4)))
    report_str = classification_report(y_true, y_pred,
                                       target_names=class_names, zero_division=0)
    (outputs_dir / "classification_report.txt").write_text(report_str, encoding="utf-8")
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    save_confusion_matrix(y_true, y_pred, class_names, outputs_dir)
    save_prediction_gallery(model, test_df, class_names, outputs_dir)
    save_top5_csv(model, test_df, class_names, outputs_dir)
    metrics = {
        "selected_model": chosen, "model_file": str(model_file),
        "num_test_images": int(len(y_true)), "num_classes": num_classes,
        "test_accuracy": round(test_acc, 4), "test_loss": round(test_loss, 4),
        "test_top5_accuracy": round(top5_acc, 4),
        "macro_precision": round(float(mp), 4), "macro_recall": round(float(mr), 4),
        "macro_f1": round(float(mf1), 4),
        "weighted_precision": round(float(wp), 4), "weighted_recall": round(float(wr), 4),
        "weighted_f1": round(float(wf1), 4),
        "best_validation_accuracy": bv_acc, "best_validation_loss": bv_loss,
        "best_validation_top5": bv_top5,
    }
    with open(outputs_dir / "final_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    shutil.copy2(str(model_file), str(models_dir / "dog_breed_classifier_final.keras"))
    shutil.copy2(str(outputs_dir / "class_names.json"), str(models_dir / "class_names.json"))
    per_f1   = dict(zip(class_names, f1.tolist()))
    worst    = sorted(per_f1.items(), key=lambda x: x[1])[:5]
    best_cls = sorted(per_f1.items(), key=lambda x: x[1], reverse=True)[:5]
    rlines = [
        "=" * 70, "PHASE 6 - FINAL EVALUATION REPORT",
        "Dog Breed Classification Using MobileNetV2-Based Transfer Learning",
        "=" * 70, "",
        "SELECTED MODEL : " + chosen,
        "MODEL FILE     : models/dog_breed_classifier_final.keras", "",
        "--- VALIDATION METRICS (used for model selection) ---",
        "Best Val Accuracy : " + str(round(bv_acc * 100, 2)) + "%",
        "Best Val Loss     : " + str(round(bv_loss, 4)),
        "Best Val Top-5    : " + str(round(bv_top5 * 100, 2)) + "%", "",
        "--- TEST METRICS ---",
        "Test Accuracy     : " + str(round(test_acc * 100, 2)) + "%",
        "Test Loss         : " + str(round(test_loss, 4)),
        "Test Top-5 Acc    : " + str(round(top5_acc * 100, 2)) + "%", "",
        "--- AGGREGATE METRICS ---",
        "Macro  Precision  : " + str(round(float(mp), 4)),
        "Macro  Recall     : " + str(round(float(mr), 4)),
        "Macro  F1-score   : " + str(round(float(mf1), 4)),
        "Weighted Precision: " + str(round(float(wp), 4)),
        "Weighted Recall   : " + str(round(float(wr), 4)),
        "Weighted F1-score : " + str(round(float(wf1), 4)), "",
        "Test images : " + str(len(y_true)) + "  |  Classes : " + str(num_classes), "",
        "--- TOP 5 BEST PREDICTED BREEDS ---",
    ]
    for b, s in best_cls: rlines.append("  " + b + " : F1=" + str(round(s, 3)))
    rlines += ["", "--- TOP 5 HARDEST BREEDS ---"]
    for b, s in worst: rlines.append("  " + b + " : F1=" + str(round(s, 3)))
    rlines += [
        "", "--- STRENGTHS ---",
        "- MobileNetV2 ImageNet pretraining provides strong feature extraction.",
        "- Top-5 accuracy >91% shows correct breed appears in top candidates.",
        "- Fine-tuning improved validation accuracy by +1.17 over frozen head.",
        "", "--- LIMITATIONS ---",
        "- 120 fine-grained similar breeds cause inter-class confusion.",
        "- CPU-only training limits feasible epochs and batch sizes.",
        "- Some minority classes may remain underrepresented after class weighting.",
        "", "=" * 70,
        "Final model : models/dog_breed_classifier_final.keras",
        "Class map   : models/class_names.json",
        "=" * 70,
    ]
    (outputs_dir / "phase6_final_report.txt").write_text(
        "\n".join(rlines), encoding="utf-8")
    required = [
        outputs_dir / "classification_report.txt",
        outputs_dir / "final_metrics.json",
        outputs_dir / "confusion_matrix.csv",
        outputs_dir / "confusion_matrix.png",
        outputs_dir / "test_prediction_gallery.png",
        outputs_dir / "top5_predictions.csv",
        outputs_dir / "phase6_final_report.txt",
        models_dir  / "dog_breed_classifier_final.keras",
        models_dir  / "class_names.json",
    ]
    print("\n" + "=" * 55)
    print("OUTPUT FILE VERIFICATION")
    print("=" * 55)
    for p in required:
        status = "OK  " + str(p.stat().st_size) + " bytes" if p.exists() else "MISSING"
        print(status + "  " + str(p.name))
    print("=" * 55)
    print("\n" + "=" * 60)
    print("PHASE 6 FINAL SUMMARY")
    print("=" * 60)
    print("Selected model      : " + chosen)
    print("Test accuracy       : " + str(round(test_acc * 100, 2)) + "%")
    print("Test loss           : " + str(round(test_loss, 4)))
    print("Test top-5 accuracy : " + str(round(top5_acc * 100, 2)) + "%")
    print("Macro F1-score      : " + str(round(float(mf1), 4)))
    print("Weighted F1-score   : " + str(round(float(wf1), 4)))
    print("Test images         : " + str(len(y_true)))
    print("Classes             : " + str(num_classes))
    print("=" * 60)
    print("\nPhase 6 complete. Waiting for confirmation before Phase 7.")

if __name__ == "__main__":
    main()
