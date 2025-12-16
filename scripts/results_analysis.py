import os

import pandas as pd
from config import config
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt

def extract_fp_df(df):
    df = df[df["Prediction"] == 1]
    df = df[df["Class"] == 0]
    return df[['ID', 'Class', 'Prediction']]

def extract_fn_df(df):
    df = df[df["Prediction"] == 0]
    df = df[df["Class"] == 1]
    return df[['ID', 'Class', 'Prediction', 'SP_15']]

def assign_tm(df, neg_df):
    merged_df = pd.merge(df, neg_df, on='ID')
    return merged_df[['ID', 'Class', 'Prediction', 'TM']]

def get_tm_fraction(df, neg_df):
    # FP dataframes
    FP_predictions = extract_fp_df(df)

    # Add TM column
    FP_predictions_w_tm = assign_tm(FP_predictions, neg_df)

    return FP_predictions_w_tm.TM.value_counts(normalize=True)[True]

def plot_sequence_logo(data, path):
    cleaned_sequences = data.str.replace('X', '-', regex=False)

    pfm_df = lm.alignment_to_matrix(
        sequences=cleaned_sequences,
        to_type='counts'  # Get raw counts of each residue at each position
    )
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

    # Calculate TM fractions
    svm_fp_tm_fraction = get_tm_fraction(svm_predictions, neg_df)
    pswm_fp_tm_fraction = get_tm_fraction(pswm_predictions, neg_df)
    # TODO plot

    # Compare sequence logos of PSWM FN
    pswm_fn = extract_fn_df(pswm_predictions)
    pos_pswm_training_set = pswm_training_set.dropna(subset=['SP_15'])
    plot_sequence_logo(pswm_fn['SP_15'], f"{config.config["results_analysis_dir"]}/fn_sp_15_sequence_logo.pdf")
    plot_sequence_logo(pos_pswm_training_set['SP_15'], f"{config.config["results_analysis_dir"]}/train_sp_15_sequence_logo.pdf")


if __name__ == "__main__":
    main()