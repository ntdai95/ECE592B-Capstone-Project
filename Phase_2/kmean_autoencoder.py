import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent
PACKET_DATA_DIR = PROJECT_ROOT / "data" / "processed_data" / "packet-data"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve)

from phase2_results import save_phase2_result

SEED = 1
K_VALUES = [2, 4, 6, 8, 10, 15, 20, 40, 50, 60, 80, 100]
KMEANS_PERCENTILES = [80, 85, 90, 92, 95, 97, 99]
BOTTLENECK_SIZES = [2, 4, 8, 16, 32, 64]
AE_PERCENTILES = [80, 85, 90, 92, 95, 97, 99]


def _select_best(results_df):
    eligible = results_df[(results_df["recall"] >= 0.90) & (results_df["fpr"] <= 0.15)]
    if eligible.empty:
        eligible = results_df[results_df["fpr"] <= 0.15]
    if eligible.empty:
        eligible = results_df
    return eligible.sort_values(["fnr", "fpr", "f1"], ascending=[True, True, False]).iloc[0]


def plot_kmeans_sweep(k_values, inertias, results_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set(title="Benign Training Inertia", xlabel="k", ylabel="Inertia")
    axes[0].grid(alpha=0.3)

    for axis, metric, title in zip(
        axes[1:],
        ["precision", "recall", "f1", "fpr", "fnr"],
        ["Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate"],
    ):
        for percentile in KMEANS_PERCENTILES:
            subset = results_df[results_df["threshold_percentile"] == percentile].sort_values("k")
            axis.plot(subset["k"], subset[metric], marker="o", label=f"Percentile {percentile}")
        axis.set(title=title, xlabel="k", ylabel="Score")
        axis.set_xticks(k_values)
        axis.grid(alpha=0.3)
        axis.legend()
    plt.tight_layout()
    plt.show()


def plot_autoencoder_sweep(bottleneck_sizes, results_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    for axis, metric, title in zip(
        axes[:5],
        ["precision", "recall", "f1", "fpr", "fnr"],
        ["Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate"],
    ):
        for percentile in AE_PERCENTILES:
            subset = results_df[results_df["threshold_percentile"] == percentile].sort_values("bottleneck_size")
            axis.plot(subset["bottleneck_size"], subset[metric], marker="o", label=f"Percentile {percentile}")
        axis.set(title=title, xlabel="Bottleneck size", ylabel="Score")
        axis.set_xticks(bottleneck_sizes)
        axis.grid(alpha=0.3)
        axis.legend()

    axes[5].axis("off")
    plt.tight_layout()
    plt.show()


def plot_final_curves(truth, kmeans_scores, ae_scores):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, scores in [("K-means", kmeans_scores), ("Autoencoder", ae_scores)]:
        fpr_curve, tpr_curve, _ = roc_curve(truth, scores)
        curve_precision, curve_recall, _ = precision_recall_curve(truth, scores)
        axes[0].plot(fpr_curve, tpr_curve, label=name)
        axes[1].plot(curve_recall, curve_precision, label=name)

    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="Test ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].set(title="Test Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(PACKET_DATA_DIR / "for_data.csv")
    df.head()

    packet_ids = df["id"].copy() if "id" in df.columns else pd.Series(df.index)
    df = df.drop(columns=["id"], errors="ignore")
    df.head()

    X = df.drop(columns=["label"]).to_numpy()
    y = df["label"].to_numpy()
    print("Data matrix shape:", X.shape)
    print("Label vector shape:", y.shape)

    row_indices = np.arange(X.shape[0])
    X_train, X_remaining, y_train, y_remaining, index_train, index_remaining = train_test_split(
        X, y, row_indices, test_size=0.4, stratify=y, random_state=SEED
    )
    X_validation, X_test, y_validation, y_test, index_validation, index_test = train_test_split(
        X_remaining,
        y_remaining,
        index_remaining,
        test_size=0.5,
        stratify=y_remaining,
        random_state=SEED,
    )

    phase2_scaler = StandardScaler()
    X_train_scaled = phase2_scaler.fit_transform(X_train)
    X_validation_scaled = phase2_scaler.transform(X_validation)
    X_test_scaled = phase2_scaler.transform(X_test)

    benign_train_mask = y_train == "benign"

    print("Train/validation/test sizes:", X_train.shape[0], X_validation.shape[0], X_test.shape[0])
    print("Benign-only training rows:", benign_train_mask.sum())
    print("Attack rows excluded from model fitting:", (~benign_train_mask).sum())
    print("Labels are retained only for analysis and model selection, not passed as features.")

    X_kmeans_train = X_train_scaled[benign_train_mask]
    X_evaluation = X_validation_scaled
    y_evaluation = y_validation
    y_attack = y_validation[y_validation != "benign"]
    true_attack = y_evaluation != "benign"

    models = {}
    thresholds = {}
    inertias = []
    kmeans_results = []

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=5,
            random_state=SEED,
        )
        model.fit(X_kmeans_train)

        training_distances = model.transform(X_kmeans_train).min(axis=1)
        evaluation_distances = model.transform(X_evaluation).min(axis=1)

        models[k] = model
        inertias.append(model.inertia_)

        for percentile in KMEANS_PERCENTILES:
            threshold = np.percentile(training_distances, percentile)
            predicted_attack = evaluation_distances > threshold

            precision = precision_score(true_attack, predicted_attack, zero_division=0)
            recall = recall_score(true_attack, predicted_attack, zero_division=0)
            f1 = f1_score(true_attack, predicted_attack, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(
                true_attack, predicted_attack, labels=[False, True]
            ).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0

            thresholds[(k, percentile)] = threshold
            kmeans_results.append({
                "k": k,
                "threshold_percentile": percentile,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "fnr": fnr,
            })

            print(
                f"k={k}, percentile={percentile}: Precision={precision:.4f}, "
                f"Recall={recall:.4f}, F1={f1:.4f}"
            )
            print("  Confusion matrix [[TP, FP], [FN, TN]]:")
            print("  Rows: Predicted [Attack, Benign]; Columns: Actual [Attack, Benign]")
            print(np.array([[tp, fp], [fn, tn]]))
            for attack in sorted(np.unique(y_attack)):
                attack_mask = y_evaluation == attack
                detection_rate = predicted_attack[attack_mask].mean()
                print(f"  {attack} detection rate: {detection_rate:.4f}")

    kmeans_results_df = pd.DataFrame(kmeans_results)
    print(
        kmeans_results_df.sort_values("fnr", ascending=True).reset_index(drop=True)
    )

    best_kmeans_config = _select_best(kmeans_results_df)
    best_k = int(best_kmeans_config["k"])
    best_kmeans_percentile = int(best_kmeans_config["threshold_percentile"])
    best_kmeans_model = models[best_k]
    best_kmeans_threshold = thresholds[(best_k, best_kmeans_percentile)]
    kmeans_test_scores = best_kmeans_model.transform(X_test_scaled).min(axis=1)
    kmeans_test_predictions = kmeans_test_scores > best_kmeans_threshold
    kmeans_test_truth = y_test != "benign"
    kmeans_test_precision = precision_score(kmeans_test_truth, kmeans_test_predictions, zero_division=0)
    kmeans_test_recall = recall_score(kmeans_test_truth, kmeans_test_predictions, zero_division=0)
    kmeans_test_f1 = f1_score(kmeans_test_truth, kmeans_test_predictions, zero_division=0)
    kmeans_tn, kmeans_fp, kmeans_fn, kmeans_tp = confusion_matrix(
        kmeans_test_truth, kmeans_test_predictions, labels=[False, True]
    ).ravel()
    kmeans_test_fpr = kmeans_fp / (kmeans_fp + kmeans_tn)
    kmeans_test_fnr = kmeans_fn / (kmeans_fn + kmeans_tp)
    kmeans_test_roc_auc = roc_auc_score(kmeans_test_truth, kmeans_test_scores)
    kmeans_test_pr_auc = average_precision_score(kmeans_test_truth, kmeans_test_scores)
    print(
        f"Final K-means test: k={best_k}, percentile={best_kmeans_percentile}, "
        f"Precision={kmeans_test_precision:.4f}, Recall={kmeans_test_recall:.4f}, "
        f"F1={kmeans_test_f1:.4f}, FPR={kmeans_test_fpr:.4f}, "
        f"FNR={kmeans_test_fnr:.4f}, ROC-AUC={kmeans_test_roc_auc:.4f}, "
        f"PR-AUC={kmeans_test_pr_auc:.4f}"
    )
    print(np.array([[kmeans_tp, kmeans_fp], [kmeans_fn, kmeans_tn]]))
    for attack in sorted(np.unique(y_test[y_test != "benign"])):
        attack_mask = y_test == attack
        print(f"  {attack} detection rate: {kmeans_test_predictions[attack_mask].mean():.4f}")

    save_phase2_result(
        "kmeans", kmeans_tn, kmeans_fp, kmeans_fn, kmeans_tp,
        kmeans_test_roc_auc, kmeans_test_pr_auc,
        {a: float(kmeans_test_predictions[y_test == a].mean())
         for a in sorted(np.unique(y_test[y_test != "benign"]))},
        k=best_k, threshold_percentile=best_kmeans_percentile,
    )

    plot_kmeans_sweep(K_VALUES, inertias, kmeans_results_df)

    keras.utils.set_random_seed(SEED)

    X_benign_train_ae = X_train_scaled[benign_train_mask].astype(np.float32)
    X_evaluation_ae = X_validation_scaled.astype(np.float32)
    y_evaluation_ae = y_validation
    y_attack_ae = y_validation[y_validation != "benign"]
    true_attack_ae = y_evaluation_ae != "benign"

    input_dim = X_benign_train_ae.shape[1]
    autoencoder_models = {}
    ae_results = []

    for bottleneck_size in BOTTLENECK_SIZES:
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)

        inputs = keras.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(64, activation="relu")(inputs)
        bottleneck = keras.layers.Dense(
            bottleneck_size, activation="relu", name="bottleneck"
        )(encoded)
        decoded = keras.layers.Dense(64, activation="relu")(bottleneck)
        outputs = keras.layers.Dense(input_dim)(decoded)
        autoencoder = keras.Model(inputs, outputs, name=f"autoencoder_{bottleneck_size}")
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        history = autoencoder.fit(
            X_benign_train_ae,
            X_benign_train_ae,
            epochs=50,
            batch_size=512,
            validation_split=0.1,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )
        autoencoder_models[bottleneck_size] = autoencoder
        print(f"Bottleneck {bottleneck_size}: trained for {len(history.history['loss'])} epochs")

        train_reconstruction = autoencoder.predict(X_benign_train_ae, verbose=0)
        train_errors = np.mean(
            (X_benign_train_ae - train_reconstruction) ** 2, axis=1
        )
        evaluation_reconstruction = autoencoder.predict(X_evaluation_ae, verbose=0)
        evaluation_errors = np.mean(
            (X_evaluation_ae - evaluation_reconstruction) ** 2, axis=1
        )

        for percentile in AE_PERCENTILES:
            threshold = np.percentile(train_errors, percentile)
            predicted_attack = evaluation_errors > threshold

            precision = precision_score(
                true_attack_ae, predicted_attack, zero_division=0
            )
            f1 = f1_score(true_attack_ae, predicted_attack, zero_division=0)
            recall = recall_score(true_attack_ae, predicted_attack, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(
                true_attack_ae, predicted_attack, labels=[False, True]
            ).ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            fnr = fn / (fn + tp) if (fn + tp) else 0.0

            ae_results.append({
                "bottleneck_size": bottleneck_size,
                "threshold_percentile": percentile,
                "threshold": threshold,
                "precision": precision,
                "f1": f1,
                "recall": recall,
                "fpr": fpr,
                "fnr": fnr,
            })

            print(
                f"bottleneck={bottleneck_size}, percentile={percentile}: "
                f"Precision={precision:.4f}, Recall={recall:.4f}, "
                f"F1={f1:.4f}"
            )
            print("  Confusion matrix [[TP, FP], [FN, TN]]:")
            print("  Rows: Predicted [Attack, Benign]; Columns: Actual [Attack, Benign]")
            print(np.array([[tp, fp], [fn, tn]]))
            for attack in sorted(np.unique(y_attack_ae)):
                attack_mask = y_evaluation_ae == attack
                detection_rate = predicted_attack[attack_mask].mean()
                print(f"  {attack} detection rate: {detection_rate:.4f}")

    ae_results_df = pd.DataFrame(ae_results)
    print(
        ae_results_df.sort_values("fnr", ascending=True).reset_index(drop=True)
    )

    best_ae_config = _select_best(ae_results_df)
    best_bottleneck_size = int(best_ae_config["bottleneck_size"])
    best_ae_percentile = int(best_ae_config["threshold_percentile"])
    best_autoencoder = autoencoder_models[best_bottleneck_size]
    best_ae_threshold = float(best_ae_config["threshold"])

    X_test_ae = X_test_scaled.astype(np.float32)
    ae_test_reconstruction = best_autoencoder.predict(X_test_ae, verbose=0)
    ae_test_scores = np.mean((X_test_ae - ae_test_reconstruction) ** 2, axis=1)
    ae_test_predictions = ae_test_scores > best_ae_threshold
    ae_test_truth = y_test != "benign"
    ae_test_precision = precision_score(ae_test_truth, ae_test_predictions, zero_division=0)
    ae_test_recall = recall_score(ae_test_truth, ae_test_predictions, zero_division=0)
    ae_test_f1 = f1_score(ae_test_truth, ae_test_predictions, zero_division=0)
    ae_tn, ae_fp, ae_fn, ae_tp = confusion_matrix(
        ae_test_truth, ae_test_predictions, labels=[False, True]
    ).ravel()
    ae_test_fpr = ae_fp / (ae_fp + ae_tn)
    ae_test_fnr = ae_fn / (ae_fn + ae_tp)
    ae_test_roc_auc = roc_auc_score(ae_test_truth, ae_test_scores)
    ae_test_pr_auc = average_precision_score(ae_test_truth, ae_test_scores)
    print(
        f"Final autoencoder test: bottleneck={best_bottleneck_size}, "
        f"percentile={best_ae_percentile}, Precision={ae_test_precision:.4f}, "
        f"Recall={ae_test_recall:.4f}, F1={ae_test_f1:.4f}, "
        f"FPR={ae_test_fpr:.4f}, FNR={ae_test_fnr:.4f}, "
        f"ROC-AUC={ae_test_roc_auc:.4f}, PR-AUC={ae_test_pr_auc:.4f}"
    )
    print(np.array([[ae_tp, ae_fp], [ae_fn, ae_tn]]))
    for attack in sorted(np.unique(y_test[y_test != "benign"])):
        attack_mask = y_test == attack
        print(f"  {attack} detection rate: {ae_test_predictions[attack_mask].mean():.4f}")

    save_phase2_result(
        "autoencoder", ae_tn, ae_fp, ae_fn, ae_tp,
        ae_test_roc_auc, ae_test_pr_auc,
        {a: float(ae_test_predictions[y_test == a].mean())
         for a in sorted(np.unique(y_test[y_test != "benign"]))},
        bottleneck_size=best_bottleneck_size, threshold_percentile=best_ae_percentile,
    )

    plot_autoencoder_sweep(BOTTLENECK_SIZES, ae_results_df)
    plot_final_curves(ae_test_truth, kmeans_test_scores, ae_test_scores)

    phase2_alert_indices = index_test[ae_test_predictions]
    phase2_alert_rows_df = pd.read_csv(PACKET_DATA_DIR / "for_data.csv").iloc[
        phase2_alert_indices
    ].copy()
    phase2_alert_rows_df["phase2_model"] = "autoencoder"
    phase2_alert_rows_df["phase2_alert"] = True
    phase2_alert_rows_df["phase2_anomaly_score"] = ae_test_scores[ae_test_predictions]
    phase2_alert_rows_df["phase2_threshold"] = best_ae_threshold
    phase2_alert_rows_df["phase2_bottleneck_size"] = best_bottleneck_size
    phase2_alert_rows_df["phase2_threshold_percentile"] = best_ae_percentile

    output_path = PACKET_DATA_DIR / "phase2_autoencoder_alert_rows_for_phase3.csv"
    phase2_alert_rows_df.to_csv(output_path, index=False)

    print(f"Exported {len(phase2_alert_rows_df)} Phase 2 autoencoder alert rows")
    print(f"Saved to: {output_path}")
    print(phase2_alert_rows_df.head())


if __name__ == "__main__":
    main()
