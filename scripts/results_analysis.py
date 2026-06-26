import os

import pandas as pd
from config import config
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt

def extract_fn_df(df):
    df = df[df["Prediction"] == 0]
    df = df[df["Class"] == 1]
    return df[['ID', 'Class', 'Prediction', 'SP_15']]

def fp_summary_row(name, df, neg_df, tm_base_rate):
    # Quantify the false positives of one model and how enriched they are in
    # transmembrane proteins relative to the negative-set base rate.
    fp = df[(df["Prediction"] == 1) & (df["Class"] == 0)]
    fn = df[(df["Prediction"] == 0) & (df["Class"] == 1)]
    tm = pd.merge(fp[["ID"]], neg_df[["ID", "TM"]], on="ID", how="left")["TM"].mean()
    return {
        "Model": name,
        "FP": len(fp),
        "FN": len(fn),
        "TM_fraction_in_FP": round(float(tm), 3),
        "TM_enrichment_vs_base": round(float(tm) / tm_base_rate, 2),
    }

def plot_sequence_logo(data, path):
    cleaned_sequences = data.str.replace('X', '-', regex=False)

    pfm_df = lm.alignment_to_matrix(
        sequences=cleaned_sequences,
        to_type='counts'  # Get raw counts of each residue at each position
    )
    # logomaker writes float probabilities back into this matrix during the
    # counts->information transform; recent pandas rejects that in-place write
    # on an integer frame, so make the counts float first.
    pfm_df = pfm_df.astype(float)
    icm_df = lm.transform_matrix(
        pfm_df,
        from_type='counts',
        to_type='information'
    )

    logo = lm.Logo(icm_df,
                   # Style options (optional)
                   color_scheme='hydrophobicity',  # A common color scheme for amino acids
                   vpad=.1,
                   width=.8)

    logo.ax.set_xlabel('Position in Alignment')
    logo.ax.set_ylabel('Information Content (bits)')
    logo.ax.set_title('Sequence Logo')

    # Set custom axis ticks for better readability (using 1-based indexing for positions)
    x_ticks = np.arange(len(icm_df))
    x_labels = x_ticks + 1
    logo.ax.set_xticks(x_ticks)
    logo.ax.set_xticklabels(x_labels)
    plt.savefig(path, dpi=1000, format="pdf",
                bbox_inches="tight",
                pad_inches=0.2)


def main():
    os.makedirs(config.config["results_analysis_dir"], exist_ok=True)

    # Load dataframes
    svm_predictions = pd.read_csv(f"{config.config['svm_dir']}/test_df_w_prediction.tsv", sep="\t")
    pswm_predictions = pd.read_csv(f"{config.config['pswm_dir']}/test_df_w_prediction.tsv", sep="\t")
    pswm_training_set = pd.read_csv(f"{config.config['pswm_dir']}/train.tsv", sep="\t")
    neg_df = pd.read_csv(f"{config.config['data_collection_dir']}/negative.tsv", sep="\t")
    neg_df.columns = ['ID', 'Taxa', 'Kingdom', 'Seq_length', 'TM']

    # Quantify false positives and their transmembrane enrichment for both models
    tm_base_rate = neg_df['TM'].mean()
    fp_summary = pd.DataFrame([
        fp_summary_row("von Heijne PSWM", pswm_predictions, neg_df, tm_base_rate),
        fp_summary_row("SVM", svm_predictions, neg_df, tm_base_rate),
    ])
    fp_summary.to_csv(
        f"{config.config['results_analysis_dir']}/fp_tm_summary.tsv",
        sep="\t", index=False,
    )
    print(f"TM base rate among negatives: {tm_base_rate*100:.1f}%")
    print(fp_summary.to_string(index=False))

    # Compare sequence logos of PSWM FN
    pswm_fn = extract_fn_df(pswm_predictions)
    pos_pswm_training_set = pswm_training_set.dropna(subset=['SP_15'])
    plot_sequence_logo(pswm_fn['SP_15'], f"{config.config["results_analysis_dir"]}/fn_sp_15_sequence_logo.pdf")
    plot_sequence_logo(pos_pswm_training_set['SP_15'], f"{config.config["results_analysis_dir"]}/train_sp_15_sequence_logo.pdf")


if __name__ == "__main__":
    main()