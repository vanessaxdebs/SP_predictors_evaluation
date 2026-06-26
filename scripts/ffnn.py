import os
import csv
import pickle, gzip

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import (precision_recall_curve, confusion_matrix, accuracy_score,
                             matthews_corrcoef, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from config import config
from scripts.svm import extract_features  # reuse the same 39 biochemical features

SEED = 42

def build_mlp(hidden_layer_sizes, alpha):
    # Feed-forward net (one or two hidden layers) on standardised features.
    # early_stopping=False so the full training loss_curve_ is available for plotting.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            alpha=alpha,
            solver="adam",
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=False,
            random_state=SEED,
        )),
    ])

def oversample_minority(X, y, seed=SEED):
    # Address class imbalance by resampling the minority class up to the majority count.
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    majority = classes[np.argmax(counts)]
    n_majority = counts.max()

    X_parts, y_parts = [X[y == majority]], [y[y == majority]]
    for c in classes:
        if c == majority:
            continue
        X_c, y_c = resample(X[y == c], y[y == c], replace=True,
                            n_samples=n_majority, random_state=seed)
        X_parts.append(X_c)
        y_parts.append(y_c)

    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)
    # shuffle so the minibatches are not ordered by class
    idx = np.random.RandomState(seed).permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]

def tune_threshold(y_true, proba):
    # Pick the decision threshold that maximises MCC on the held-out predictions.
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    best_thr, best_mcc = 0.5, -1.0
    for thr in thresholds:
        mcc = matthews_corrcoef(y_true, (proba >= thr).astype(int))
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
    return best_thr

def metrics_row(y_true, y_pred):
    return [accuracy_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            matthews_corrcoef(y_true, y_pred)]

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm_ = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm_.ravel()
    plot_cm = np.array([[TP, FP], [FN, TN]])
    cell_colors = np.array([['#fde725ff', '#483677ff'],
                            ['#2d708eff', '#3cbb75ff']])

    fig, ax = plt.subplots(figsize=(5, 4))
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=cell_colors[r, c], linewidth=0))
            text_color = 'black' if (r, c) == (0, 0) else 'white'
            ax.text(c + 0.5, 1.5 - r, plot_cm[r, c], ha='center', va='center',
                    fontsize=12, fontfamily='Liberation Serif', color=text_color)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(['True Positive', 'True Negative'])
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Predicted Negative', 'Predicted Positive'], rotation=90, va='center')
    ax.set_title(f'Confusion Matrix: {title}', fontsize=14, fontfamily='Liberation Serif', pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

def main():
    os.makedirs(config.config["ffnn_dir"], exist_ok=True)
    out = config.config["ffnn_dir"]

    sns.set_theme()
    mpl.rcParams['font.family'] = ['Liberation Serif', 'serif']
    mpl.rcParams['axes.unicode_minus'] = False
    sns.set_theme(context='notebook', style='white', palette='viridis', font='Liberation Serif', font_scale=1.1)

    df = pd.read_csv(f"{config.config['pswm_dir']}/train.tsv", sep="\t")
    test_df = pd.read_csv(f"{config.config['pswm_dir']}/test.tsv", sep="\t")

    # Extract the features once for the whole train and test sets, then index by fold.
    X_all = extract_features(df['Frag_90'])
    y_all = df['Class'].to_numpy()
    groups = df['Group'].to_numpy()
    X_test = extract_features(test_df['Frag_90'])
    y_test = test_df['Class'].to_numpy()

    cycles = 5
    grid = [
        {"hidden_layer_sizes": (64,), "alpha": 1e-4},
        {"hidden_layer_sizes": (64,), "alpha": 1e-3},
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-3},
        {"hidden_layer_sizes": (32, 16), "alpha": 1e-3},
    ]

    # 5-fold cross validation over the architecture grid, selecting by mean validation MCC.
    results = {}  # config index -> per-fold dict
    for gi, params in enumerate(grid):
        fold_metrics, oof_y, oof_proba, loss_curves = [], [], [], []
        for k in range(cycles):
            tr = groups != (k + 1)
            va = groups == (k + 1)
            X_bal, y_bal = oversample_minority(X_all[tr], y_all[tr])

            model = build_mlp(params["hidden_layer_sizes"], params["alpha"])
            model.fit(X_bal, y_bal)

            proba = model.predict_proba(X_all[va])[:, 1]
            thr = tune_threshold(y_all[va], proba)
            pred = (proba >= thr).astype(int)

            fold_metrics.append(metrics_row(y_all[va], pred) + [thr])
            oof_y.append(y_all[va]); oof_proba.append(proba)
            loss_curves.append(model.named_steps["mlp"].loss_curve_)

        mean_mcc = np.mean([m[4] for m in fold_metrics])
        results[gi] = {
            "params": params, "mean_mcc": mean_mcc, "fold_metrics": fold_metrics,
            "oof_y": np.concatenate(oof_y), "oof_proba": np.concatenate(oof_proba),
            "loss_curves": loss_curves,
        }
        print(f"FFNN config {gi} {params} -> mean validation MCC {mean_mcc:.3f}")

    best_gi = max(results, key=lambda g: results[g]["mean_mcc"])
    best = results[best_gi]
    print(f"\nSelected FFNN architecture: {best['params']} (mean val MCC {best['mean_mcc']:.3f})")

    # Save the per-fold metrics of the selected architecture.
    cv_df = pd.DataFrame(best["fold_metrics"],
                         columns=["Accuracy", "Recall", "Precision", "F1 Score", "MCC", "Threshold"])
    cv_df.to_csv(f"{out}/cv_metrics.tsv", sep="\t", quoting=csv.QUOTE_NONE)

    # Training curves (training loss per epoch) for each fold of the selected architecture.
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, cycles))
    for k, curve in enumerate(best["loss_curves"]):
        plt.plot(range(1, len(curve) + 1), curve, color=colors[k], label=f"Fold {k + 1}")
    plt.xlabel("Epoch"); plt.ylabel("Training loss (log-loss)")
    plt.title("FFNN Training Curves (selected architecture)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{out}/training_curves.pdf", dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

    # Decision threshold tuned on the pooled out-of-fold predictions (no test leakage).
    final_threshold = tune_threshold(best["oof_y"], best["oof_proba"])

    # Final model trained on the whole (balanced) training set with the selected architecture.
    X_bal, y_bal = oversample_minority(X_all, y_all)
    final_model = build_mlp(best["params"]["hidden_layer_sizes"], best["params"]["alpha"])
    final_model.fit(X_bal, y_bal)

    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= final_threshold).astype(int)

    # Persist the model artifact together with its scaler, architecture and threshold.
    pickle.dump(
        {"model": final_model, "threshold": final_threshold, "params": best["params"]},
        gzip.open(f"{out}/model.pkl.gz", "w"),
    )

    test_df_w_prediction = test_df.copy()
    test_df_w_prediction["Probability"] = test_proba
    test_df_w_prediction["Prediction"] = test_pred
    test_df_w_prediction.to_csv(f"{out}/test_df_w_prediction.tsv", sep="\t",
                                quoting=csv.QUOTE_NONE, index=False)

    test_metrics = metrics_row(y_test, test_pred)  # [Acc, Rec, Prec, F1, MCC]
    pd.DataFrame(
        [["Accuracy", test_metrics[0]], ["MCC", test_metrics[4]], ["Recall", test_metrics[1]],
         ["Precision", test_metrics[2]], ["F1 score", test_metrics[3]]],
        columns=["Metric", "Value"],
    ).to_csv(f"{out}/test_df_metrics.tsv", sep="\t", quoting=csv.QUOTE_NONE)

    plot_confusion_matrix(y_test, test_pred, "FFNN", f"{out}/confusion_matrix.pdf")
    print(f"\nFFNN test metrics — Acc {test_metrics[0]:.3f}, Precision {test_metrics[2]:.3f}, "
          f"Recall {test_metrics[1]:.3f}, F1 {test_metrics[3]:.3f}, MCC {test_metrics[4]:.3f} "
          f"(threshold {final_threshold:.3f})")

if __name__ == "__main__":
    main()
