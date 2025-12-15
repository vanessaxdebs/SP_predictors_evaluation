import pandas as pd

from config import config

def extract_fp_df(df):
    df = df[df["Prediction"] == 1]
    df = df[df["Class"] == 0]
    return df[['ID', 'Class', 'Prediction']]

def assign_tm(df, neg_df):
    merged_df = pd.merge(df, neg_df, on='ID')
    return merged_df[['ID', 'Class', 'Prediction', 'TM']]

def get_tm_fraction(df, neg_df):
    # FP dataframes
    FP_predictions = extract_fp_df(df)

    # Add TM column
    FP_predictions_w_tm = assign_tm(FP_predictions, neg_df)

    return FP_predictions_w_tm.TM.value_counts(normalize=True)[True]


if __name__ == "__main__":
    # Load dataframes
    svm_predictions = pd.read_csv(f"{config.config['svm_dir']}/test_df_w_prediction.tsv", sep="\t")
    pswm_predictions = pd.read_csv(f"{config.config['pswm_dir']}/test_df_w_prediction.tsv", sep="\t")
    neg_df = pd.read_csv(f"{config.config['data_collection_dir']}/negative.tsv", sep="\t")
    neg_df.columns = ['ID', 'Taxa', 'Kingdom', 'Seq_length', 'TM']

    # Calculate TM fractions
    svm_fp_tm_fraction = get_tm_fraction(svm_predictions, neg_df)
    pswm_fp_tm_fraction = get_tm_fraction(pswm_predictions, neg_df)
