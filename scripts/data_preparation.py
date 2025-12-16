import os
import subprocess
import pandas as pd
import csv
from config import config

def exec_mmseqs_easy_cluster(
        input_dir,
        output_dir,
        prefix,
        min_seq_id,
        coverage_threshold,
):
    subprocess.run(f"rm -rf {output_dir}/{prefix} \
                    && mkdir {output_dir}/{prefix} \
                    && cd {output_dir}/{prefix} \
                    && mmseqs easy-cluster {input_dir}/{prefix}.fasta cluster-results output --min-seq-id {min_seq_id} -c {coverage_threshold} --cov-mode 0 --cluster-mode 1 \
                    && cd ..",
                   shell=True,
                   text=True)

def filter_redundant_seq(
    clustered_tsv_path,
    seq_tsv_path,
    save_path,
):
    """
    filter_redundant_seq creates a dataframe with only representative sequences
    :param clustered_tsv_path: path to tsv with clustered sequences
    :param seq_tsv_path: path to tsv with sequences
    :param save_path: path to save filtered tsv
    """
    clusters_df = pd.read_csv(clustered_tsv_path, sep='\t', names=['cluster', 'sequence'])
    reps = clusters_df['cluster'].unique()

    df = pd.read_csv(seq_tsv_path, sep='\t', names=['ID', 'Taxa', 'Kingdom', 'Seq_Length', 'Site'])

    filtered = df.loc[df['ID'].isin(reps)]
    filtered.to_csv(save_path, sep='\t', quoting=csv.QUOTE_NONE)

    return filtered

def train_test_validation_split(df):
    i = int(len(df) * 0.8)

    train = df.iloc[0:i]
    test = df.iloc[i:]
    test = test.assign(Group = 0)
    train = train.assign(Group=100)

    l = [0, int(len(train) * 0.2), int(len(train) * 0.4), int(len(train) * 0.6), int(len(train) * 0.8), len(train)]

    validation_sets = []
    for n in range(len(l) - 1):
        validation_sets.append(train.iloc[l[n]:l[n + 1]])
        train["Group"][l[n]:l[n + 1]] = n + 1


    return train, test, validation_sets


def main():
    os.makedirs(config.config["data_preparation_dir"], exist_ok=True)

    exec_mmseqs_easy_cluster(
        config.config["data_collection_dir_inside_output"],
        config.config["data_preparation_dir"],
        config.config["positive_prefix"],
        config.config["min_seq_id"],
        config.config["coverage_threshold"],
    )

    pos_filtered = filter_redundant_seq(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}/cluster-results_cluster.tsv",
        f"{config.config["data_collection_dir"]}/{config.config["positive_prefix"]}.tsv",
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_filtered.tsv"
    )

    pos_train, pos_test, validation_sets = train_test_validation_split(pos_filtered)
    pos_test.to_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_test.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    pos_train.to_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_train.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    for i in range(len(validation_sets)):
        print(validation_sets[i])
        validation_sets[i].to_csv(
            f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_train_{i}.tsv",
            sep='\t',
            quoting=csv.QUOTE_NONE,
        )

    exec_mmseqs_easy_cluster(
        config.config["data_collection_dir_inside_output"],
        config.config["data_preparation_dir"],
        config.config["negative_prefix"],
        config.config["min_seq_id"],
        config.config["coverage_threshold"],
    )

    neg_filtered = filter_redundant_seq(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}/cluster-results_cluster.tsv",
        f"{config.config["data_collection_dir"]}/{config.config["negative_prefix"]}.tsv",
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_filtered.tsv"
    )

    neg_train, neg_test, validation_sets = train_test_validation_split(neg_filtered)
    neg_test.to_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_test.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )
    neg_train.to_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_train.tsv",
        sep='\t',
        quoting=csv.QUOTE_NONE,
    )

    for i in range(len(validation_sets)):
        validation_sets[i].to_csv(
            f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_train_{i}.tsv",
            sep='\t',
            quoting=csv.QUOTE_NONE,
        )

if __name__ == "__main__":
    main()