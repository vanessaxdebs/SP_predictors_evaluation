import csv
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from config import config

def compute_matrix(
        frag_15:pd.Series,
        output_arrayname:str,
) -> np.ndarray:
  swissprot = np.array([0.0825, 0.0393, 0.0964, 0.0665,0.0552, 0.0671, 0.0579, 0.0536, 0.0406, 0.0707, 0.0410, 0.0100,
                      0.0546, 0.0227, 0.0386, 0.0292, 0.0138, 0.0590, 0.0474, 0.0685])
  alphabet = "AQLSREKTNGMWDHFYCIPV"

  pswm = np.ones((len(alphabet),len(frag_15[0])))

  seq_number = 0
  for frag in frag_15:
      seq_number += 1
      frag_m = np.zeros((len(alphabet),len(frag)))
      col = 0
      for res in frag:
        if res != "X":
          index = alphabet.index(res)
          frag_m[index][col] += 1
          col += 1
      pswm += frag_m
  pswm = (pswm/(seq_number+20))
  pswm = pswm / swissprot.reshape(-1, 1)
  pswm = np.log2(pswm)
  np.savez(output_arrayname, pswm)
  return pswm

def compute_score(sequence,  window, pswm, alphabet = "AQLSREKTNGMWDHFYCIPV"):
    seq_len = len(sequence)
    scores = np.zeros(seq_len - window + 1)

    for pos in range(seq_len - window + 1):
        window_seq = sequence[pos:pos+window]
        score = 0.0
        for i, res in enumerate(window_seq):
            if res != "X":
                idx = alphabet.find(res)
                score += pswm[idx, i]
        scores[pos] = score

    return scores.max()

def calc_threshold(
        df,
        scores,
        save_path,
):
    classes = df['Class']
    precision, recall, thresholds = precision_recall_curve(classes, scores)
    fscore = (2 * precision * recall) / (precision + recall + 1e-15)

    index = np.argmax(fscore)
    optimal_threshold = thresholds[index]

    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot()

    optimal_precision = precision[index]
    optimal_recall = recall[index]
    optimal_fscore = fscore[index]

    plt.scatter(optimal_recall, optimal_precision, color='yellow', s=100, label=f'Optimal (F1={optimal_fscore:.3f})',
                zorder=5)
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

    return optimal_threshold, optimal_fscore, optimal_precision, optimal_recall

def predict(
    df,
    pwsm,
    optimal_threshold,
):
    df = df.assign(Prediction=None)

    for index, row in df.iterrows():
        score = compute_score(row['Frag_90'], 15, pwsm)
        if score >= optimal_threshold:
            df.loc[index, 'Prediction'] = 1
        else:
            df.loc[index, 'Prediction'] = 0


    return df

def compute_confusion_matrix(df):
  TP, TN, FP, FN = 0,0,0,0
  for i, row in df.iterrows():
    if row['Prediction'] == row['Class'] and row['Class'] == 0:
      TN += 1
    elif row['Prediction'] == row['Class'] and row['Class'] == 1:
      TP += 1
    elif row['Prediction'] != row['Class'] and row['Class'] == 0:
      FP += 1
    elif row['Prediction'] != row['Class'] and row['Class'] == 1:
      FN +=1
  return TP, FP, FN, TN

def accuracy(TP, FP, FN, TN):
    return (TP + TN) / (TP + TN + FP + FN)

def mcc(TP, FP, FN, TN):
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0.0

def recall(TP, FP, FN, TN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0.0

def precision(TP, FP, FN, TN):
    return TP / (TP + FP) if (TP + FP) != 0 else 0.0

def f1_score(prec, rec):
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.0

if __name__ == "__main__":
    # Set theme in Seaborn
    sns.set_theme()
    mpl.rcParams['font.family'] = ['Liberation Serif', 'serif']
    mpl.rcParams['font.serif'] = ['Liberation Serif']
    mpl.rcParams['axes.unicode_minus'] = False  # avoid minus rendering issues
    sns.set_theme(context='notebook', style='white', palette='viridis', font='Liberation Serif', font_scale=1.1)

    df = pd.read_csv(f"{config.config['pswm_dir']}/train.tsv", sep = "\t")

    # Compute matrix
    group_mask = df['Group'].isin([1,2,3])
    class_mask = df['Class'] == 1
    combined_mask = group_mask & class_mask
    filtered_df = df[combined_mask].copy()
    pwsm = compute_matrix(filtered_df['SP_15'], f"{config.config['pswm_dir']}/matrix")

    # Compute scores and threshold
    group_4 = df[df['Group'] == 4]
    sequences = group_4['Frag_90']
    scores = np.array([])
    for seq in sequences:
        score = compute_score(seq, 15, pwsm)
        scores = np.append(scores, score)

    threshold, opt_fscore, opt_precision, opt_recall = calc_threshold(group_4, scores, f"{config.config['pswm_dir']}/threshold.pdf")

    # Predict
    df_w_prediction = predict(df[df['Group'] == 5], pwsm, threshold)
    df_w_prediction[['ID', 'Class', 'Prediction', 'Seq_Length', 'Site', 'Group', 'Frag_90', 'SP_15']].to_csv(
        f"{config.config['pswm_dir']}/prediction.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    # Calculate metrics
    metrics_data = []
    TP, FP, FN, TN = compute_confusion_matrix(df_w_prediction)
    metrics_data.append(['Accuracy', accuracy(TP, FP, FN, TN)])
    metrics_data.append(['MCC', mcc(TP, FP, FN, TN)])
    recall_value = recall(TP, FP, FN, TN)
    metrics_data.append(['Recall', recall_value])
    precision_value = precision(TP, FP, FN, TN)
    metrics_data.append(['Precision', precision_value])
    metrics_data.append(['F1 score', f1_score(precision_value, recall_value)])
    pd.DataFrame(metrics_data, columns=["Metric", "Value"]).to_csv(
        f"{config.config['pswm_dir']}/metrics.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    # Cross validation
    test_df = pd.read_csv(f"{config.config['pswm_dir']}/test.tsv", sep = "\t")

    cycles = 5
    accuracies = np.array([])
    precisions = np.array([])
    recalls = np.array([])
    f1s = np.array([])
    mccs = np.array([])
    ths = np.array([])

    for i in range(cycles):
        training_df = df[df["Group"] != i+1]
        validation_df = df[df["Group"] == i+1]

        filtered_df = training_df[training_df["Class"] == 1].copy()
        filtered_df.reset_index(drop=True, inplace=True)
        pswm = compute_matrix(filtered_df['SP_15'], f"{config.config['pswm_dir']}/matrix_cv_{i+1}")

        # Create the PSWM heatmap
        alphabet = list("AQLSREKTNGMWDHFYCIPV")

        plt.figure(figsize=(10, 6))
        sns.heatmap(pswm, cmap="viridis", annot=True, fmt=".2f", yticklabels=alphabet,
                    xticklabels=range(1, pswm.shape[1] + 1), cbar_kws={'label': 'Score (log2 ratio)'})

        plt.title(f"Position-Specific Weight Matrix (PSWM {i + 1})", fontsize=14)
        plt.xlabel("Position (15-mer window)", fontsize=12)
        plt.ylabel("Amino Acid", fontsize=12)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{config.config['pswm_dir']}/PSWM_cv_{i+1}.pdf", dpi=1000, format="pdf",bbox_inches="tight",pad_inches=0.2)
        plt.clf()

        # validation
        sequences = validation_df['Frag_90']
        validation_scores = np.array([])
        for seq in sequences:
            score = compute_score(seq, 15, pwsm)
            validation_scores = np.append(validation_scores, score)

        threshold, opt_fscore, opt_precision, opt_recall = calc_threshold(validation_df, validation_scores,
                                                                          f"{config.config['pswm_dir']}/threshold_cv_{i+1}.pdf")


        # Calculate metrics
        df_val_prediction = predict(validation_df, pwsm, threshold)
        TP, FP, FN, TN = compute_confusion_matrix(df_val_prediction)

        acc = accuracy(TP, FP, FN, TN)
        rec = recall(TP, FP, FN, TN)
        prec = precision(TP, FP, FN, TN)
        f1 = f1_score(prec, rec)
        MCC = mcc(TP, FP, FN, TN)

        accuracies = np.append(accuracies, acc)
        recalls = np.append(recalls, rec)
        precisions = np.append(precisions, prec)
        f1s = np.append(f1s, f1)
        mccs = np.append(mccs, MCC)
        ths = np.append(ths, threshold)

    cv_metrics_df = pd.DataFrame(
        {
            'Accuracy': accuracies,
            'Recall': recalls,
            'Precision': precisions,
            'F1 Score': f1s,
            'MCC': mccs,
            'Threshold': ths,
        },
    )

    cv_metrics_df.to_csv(
        f"{config.config['pswm_dir']}/cv_metrics.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    # We pick the threshold from the iteration with the highest f1 score and do testing
    iter_id = cv_metrics_df['F1 Score'].idxmax()
    best_threshold = cv_metrics_df.iloc[cv_metrics_df['F1 Score'].idxmax()]["Threshold"]
    best_pswm = np.load(f"{config.config['pswm_dir']}/matrix_cv_{iter_id}.npz")

    # testing
    test_df_w_prediction = predict(test_df, best_pswm['arr_0'], best_threshold)
    test_df_w_prediction[['ID', 'Class', 'Prediction', 'Seq_Length', 'Site', 'Group', 'Frag_90', 'SP_15']].to_csv(
        f"{config.config['pswm_dir']}/test_df_w_prediction.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    metrics_data = []
    TP, FP, FN, TN = compute_confusion_matrix(test_df_w_prediction)
    acc = accuracy(TP, FP, FN, TN)
    metrics_data.append(['Accuracy', acc])
    mcc_score = mcc(TP, FP, FN, TN)
    metrics_data.append(['MCC', mcc_score])
    recall_value = recall(TP, FP, FN, TN)
    metrics_data.append(['Recall', recall_value])
    precision_value = precision(TP, FP, FN, TN)
    metrics_data.append(['Precision', precision_value])
    f1_score_value = f1_score(precision_value, recall_value)
    metrics_data.append(['F1 score', f1_score_value])
    pd.DataFrame(metrics_data, columns=["Metric", "Value"]).to_csv(
        f"{config.config['pswm_dir']}/test_df_metrics.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    # Confusion Matrix
    plot_cm = np.array([[TP, FP],
                        [FN, TN]])

    cell_colors = np.array([['#fde725ff', '#483677ff'],
                            ['#2d708eff', '#3cbb75ff']])

    fig, ax = plt.subplots(figsize=(5, 4))

    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, color=cell_colors[r, c], edgecolor=None, linewidth=0))
            text_color = 'black' if (r, c) == (0, 0) else 'white'
            ax.text(c + 0.5, 1.5 - r, plot_cm[r, c],
                    ha='center', va='center', fontsize=12,
                    fontfamily='Liberation Serif', color=text_color)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['True Positive', 'True Negative'], fontname='Liberation Serif')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Predicted Negative', 'Predicted Positive'], rotation=90, va='center',
                       fontname='Liberation Serif')

    ax.set_title(f'Confusion Matrix\nAcc={acc:.3f}, F1={f1_score_value:.3f}, MCC={mcc_score:.3f}', fontsize=14,
                 fontfamily='Liberation Serif', pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.config['pswm_dir']}/confusion_matrix.pdf", dpi=1000, format="pdf", bbox_inches="tight",
                pad_inches=0.2)
    plt.clf()


