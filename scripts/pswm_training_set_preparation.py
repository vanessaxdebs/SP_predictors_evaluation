import os
import subprocess
import pandas as pd
from Bio import SeqIO
from typing import Dict, List, Any
from config import config

def merge_fasta_files(input_files: List[str], output_file: str):
    """
    Merges multiple FASTA files into a single output file using the 'cat' command.

    Args:
        input_files: A list of file paths for the FASTA files to be merged.
        output_file: The path for the single merged output FASTA file.
    """
    command_args = ['cat'] + input_files

    print(f"Executing command: {' '.join(command_args)} > {output_file}")

    with open(output_file, 'w') as outfile:
        subprocess.run(
            command_args,
            check=True,
            stdout=outfile,
            shell=False
        )

    print(f"Successfully merged {len(input_files)} files into {output_file}")

def add_sequences_from_fasta(
        df: pd.DataFrame,
        fasta_file_path: str,
        id_column: str = "ID"
) -> pd.DataFrame:
    print(f"Loading sequences from {fasta_file_path}...")

    sequence_dict: Dict[str, str] = {}

    for record in SeqIO.parse(fasta_file_path, "fasta"):
        sequence_dict[record.id] = str(record.seq)

    print(f"Found {len(sequence_dict)} sequences in the FASTA file.")

    df['Sequence'] = df[id_column].map(sequence_dict)

    missing_count = df['Sequence'].isna().sum()
    if missing_count > 0:
        print(f"\nWarning: {missing_count} sequences could not be found in the FASTA file.")

    print("\n'Sequence' column successfully added to the DataFrame.")

    return df


def add_fragment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Frag_90 and SP_15 columns to the dataframe based on the sequence
    and the cleavage site position.
    """

    # 1. Add Frag_90: First 90 characters
    # Python slicing [0:90] takes characters from index 0 up to (but not including) 90
    df['Frag_90'] = df['Sequence'].str[:90]

    # 2. Add SP_15: [Site-13 to Site+2]
    # Assuming 'Site' is 1-based (e.g., if Site is 20, character 20 is at index 19)
    # Target: First char = Site - 13 (1-based) -> Index = (Site - 13) - 1
    # Target: Last char = Site + 2 (1-based)  -> Index = (Site + 2)

    # We use .apply because the slice indices change for every row
    def extract_sp15(row):
        try:
            site = int(row['Site'])
            start_idx = (site - 13) - 1
            end_idx = (site + 2)  # In slicing [start:end], 'end' is exclusive

            # Handle potential negative start indices if Site < 14
            if start_idx < 0:
                return "None"

            return row['Sequence'][start_idx:end_idx]
        except (ValueError, TypeError):
            return "None"

    df['SP_15'] = df.apply(extract_sp15, axis=1)

    return df

if __name__ == "__main__":
    os.makedirs(config.config["pswm_dir"], exist_ok=True)

    train_pos_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_train.tsv",
        sep='\t',
    )
    train_neg_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_train.tsv",
        sep='\t',
    )

    train_pos_df = train_pos_df.assign(Class=1)
    train_neg_df = train_neg_df.assign(Class=0)

    train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True, sort=False)

    # Keep only useful features
    train_df = train_df[["ID", "Seq_Length", "Site", "Class", "Group"]]

    # Merge .fasta into a single file
    merge_fasta_files(
        [
            f"{config.config["data_collection_dir"]}/negative.fasta",
            f"{config.config["data_collection_dir"]}/positive.fasta",
        ],
        f"{config.config['pswm_dir']}/train.fasta",
    )

    # Enrich df with fasta
    train_df = add_sequences_from_fasta(train_df, f"{config.config['pswm_dir']}/train.fasta")
    train_df = add_fragment_columns(train_df)
    train_df = train_df.drop('Sequence', axis=1)
    train_df.to_csv(f"{config.config['pswm_dir']}/train.tsv", sep='\t')


    # Prepare also testing df here
    test_pos_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_test.tsv",
        sep='\t',
    )
    test_neg_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_test.tsv",
        sep='\t',
    )

    test_pos_df = test_pos_df.assign(Class=1)
    test_neg_df = test_neg_df.assign(Class=0)

    test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True, sort=False)

    # Keep only useful features
    test_df = test_df[["ID", "Seq_Length", "Site", "Class", "Group"]]

    # Enrich df with fasta
    test_df = add_sequences_from_fasta(test_df, f"{config.config['pswm_dir']}/train.fasta")
    test_df = add_fragment_columns(test_df)
    test_df = test_df.drop('Sequence', axis=1)
    test_df.to_csv(f"{config.config['pswm_dir']}/test.tsv", sep='\t')