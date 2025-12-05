import os

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from config import config

def dist_by_kingdom_plot(
        df,
        label,
        save_path,
):
    # Count the values and percentages
    t_kingdom = df['Kingdom'].value_counts()
    t_percent = t_kingdom / t_kingdom.sum() * 100

    # Divide the plot in two
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(label, fontsize=16, fontweight='bold')

    # Create the pie plot and format the labels
    wedges, texts, autotexts = ax1.pie(t_kingdom, labels=t_kingdom.index, autopct='%.1f%%', )
    ax1.axis('equal')

    for text in texts:
        text.set_color("black")
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # Create the bar plot
    sns.barplot(hue=t_percent.index, y=t_percent.values, legend=True, ax=ax2, palette="viridis", native_scale=True)
    ax2.set_ylabel("Percentage (%)")

    # Display the plots
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=1000, format="pdf")
    plt.clf()

def dist_by_species_plot(
    df,
    label,
    save_path,
):
    # Edit the dataframe to preserve only the most frequent species.
    keep_labels = ["Homo sapiens", "Mus musculus", "Saccharomyces cerevisiae (strain ATCC 204508 / S288c)",
                   "Arabidopsis thaliana", "Schizosaccharomyces pombe (strain 792 / ATCC 24843)",
                   "Drosophila melanogaster"]
    for index, row in df.iterrows():
        if row["Taxa"] not in keep_labels:
            df.at[index, "Taxa"] = "Other"

    # Count the values and locate 'Other' as the last section of the pie
    t_species = df['Taxa'].value_counts()
    if "Other" in t_species.index:
        t_species = pd.concat([t_species.drop("Other"), t_species.loc[["Other"]]])

    t_percent = t_species / t_species.sum() * 100

        # Divide the plot in two
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(label, fontsize=16, fontweight='bold')

    # Create the pie plot and format the labels
    wedges, texts, autotexts = ax1.pie(t_species, labels=t_species.index, autopct='%.1f%%', )
    ax1.axis('equal')

    for text in texts:
        text.set_color("black")
        text.set_fontsize(11)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # Create the bar plot
    sns.barplot(hue=t_percent.index, y=t_percent.values, legend=True, ax=ax2, palette="viridis", native_scale=True)
    ax2.set_ylabel("Percentage (%)")

    # Save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=1000, format="pdf",bbox_inches="tight",pad_inches=0.2)
    plt.clf()

def seq_len_dist_plot(
    df,
    hue,
    title,
    labels,
    save_path,
):
    d = sns.kdeplot(data=df, x="Seq_Length", hue=hue, common_norm=False, palette="viridis", fill=True)
    d.set(title="Sequence Length Distributions", xlabel="Sequence Length", xlim=(0, 5000))
    sns.move_legend(d, "upper right", title=title, labels=labels)
    d.set(xlim=(0, 5000))
    plt.savefig(save_path, dpi=1000, format="pdf",bbox_inches="tight",pad_inches=0.2)
    plt.clf()

def seq_len_boxplot(
    df,
    hue,
    title,
    labels,
    save_path,
):
    a = sns.boxplot(data=df, y="Seq_Length", hue=hue, width=0.6, palette="viridis", fill=False)
    a.set(title="Sequence Length Distributions", ylabel="Sequence Length", ylim=(0, 4000))
    sns.move_legend(a, "upper right", title=title, labels=labels)
    plt.savefig(save_path, dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

def cleavage_len_boxplot(
    df,
    hue,
    title,
    labels,
    save_path,
):
    a = sns.boxplot(data=df, y="Site", hue=hue, width=0.6, palette="viridis", fill=False)
    a.set(title="SP Length Distributions", ylabel="SP Length", ylim=(0,70))
    sns.move_legend(a, "upper right", title=title, labels=labels)
    plt.savefig(save_path, dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

def cleavage_len_his(
    df,
    hue,
    title,
    labels,
    save_path,
):
    a = sns.displot(data=df, x="Site", hue=hue, common_norm=False, stat="probability", palette="viridis", fill=True)
    a.set(title="SP Length Distributions", xlabel="SP Length", xlim=(0, 70))
    sns.move_legend(a, "center", bbox_to_anchor=(1, 0.6), title=title, labels=labels)
    plt.savefig(save_path, dpi=1000, format="pdf", bbox_inches="tight", pad_inches=0.2)
    plt.clf()

if __name__ == "__main__":
    os.makedirs(config.config["data_analysis_dir"], exist_ok=True)

    # Set up theme
    sns.set_theme(context='notebook', style='whitegrid', palette='viridis', font='Times New Roman', font_scale=1,
                  color_codes=True, rc={"font.family": ["Times New Roman", "Times", "Times Roman", "serif"]})
    mpl.rcParams['font.family'] = 'Times New Roman'

    # Download datasets
    test_pos_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_test.tsv",
        sep='\t',
    )
    train_pos_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["positive_prefix"]}_train.tsv",
        sep='\t',
    )
    test_neg_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_test.tsv",
        sep='\t',
    )
    train_neg_df = pd.read_csv(
        f"{config.config["data_preparation_dir"]}/{config.config["negative_prefix"]}_train.tsv",
        sep='\t',
    )
    train_df = pd.concat([train_pos_df, train_neg_df], ignore_index=True, sort=False)
    test_df = pd.concat([test_pos_df, test_neg_df], ignore_index=True, sort=False)

    # Analysis
    dist_by_kingdom_plot(
        train_df,
        "Training Set",
        f"{config.config["data_analysis_dir"]}/train_kingdoms.pdf",
    )
    dist_by_kingdom_plot(
        train_pos_df,
        "Positive Training Set",
        f"{config.config["data_analysis_dir"]}/pos_train_kingdoms.pdf",
    )
    dist_by_kingdom_plot(
        train_neg_df,
        "Negative Training Set",
        f"{config.config["data_analysis_dir"]}/neg_train_kingdoms.pdf",
    )
    dist_by_kingdom_plot(
        test_df,
        "Testing Set",
        f"{config.config["data_analysis_dir"]}/test_kingdoms.pdf",
    )
    dist_by_kingdom_plot(
        test_pos_df,
        "Positive Testing Set",
        f"{config.config["data_analysis_dir"]}/pos_test_kingdoms.pdf",
    )
    dist_by_kingdom_plot(
        test_neg_df,
        "Negative Testing Set",
        f"{config.config["data_analysis_dir"]}/neg_test_kingdoms.pdf",
    )

    dist_by_species_plot(
        train_df,
        "Training Set",
        f"{config.config["data_analysis_dir"]}/train_species.pdf",
    )
    dist_by_species_plot(
        train_pos_df,
        "Positive Training Set",
        f"{config.config["data_analysis_dir"]}/pos_train_species.pdf",
    )
    dist_by_species_plot(
        train_neg_df,
        "Negative Training Set",
        f"{config.config["data_analysis_dir"]}/neg_train_species.pdf",
    )
    dist_by_species_plot(
        test_df,
        "Testing Set",
        f"{config.config["data_analysis_dir"]}/test_species.pdf",
    )
    dist_by_species_plot(
        test_pos_df,
        "Positive Testing Set",
        f"{config.config["data_analysis_dir"]}/pos_test_species.pdf",
    )
    dist_by_species_plot(
        test_neg_df,
        "Negative Testing Set",
        f"{config.config["data_analysis_dir"]}/neg_test_species.pdf",
    )

    # Enrich DFs with metadata
    train_neg_df_sp = train_neg_df.assign(SP=0)
    train_pos_df_sp = train_pos_df.assign(SP=1)
    test_neg_df_sp = test_neg_df.assign(SP=0)
    test_pos_df_sp = test_pos_df.assign(SP=1)
    train_neg_df_sp_and_t = train_neg_df_sp.assign(Training=1)
    train_pos_df_sp_ant_t = train_pos_df_sp.assign(Training=1)
    test_neg_df_sp_and_t = test_neg_df_sp.assign(Training=0)
    test_pos_df_sp_ant_t = test_pos_df_sp.assign(Training=0)

    train_df_with_md = pd.concat([train_neg_df_sp_and_t, train_pos_df_sp_ant_t], ignore_index=True, sort=False)
    test_df_with_md = pd.concat([test_neg_df_sp_and_t, test_pos_df_sp_ant_t], ignore_index=True, sort=False)
    df_with_md = pd.concat([train_df_with_md, test_df_with_md], ignore_index=True, sort=False)
    positive_df_with_md = pd.concat([train_pos_df_sp_ant_t, test_pos_df_sp_ant_t], ignore_index=True, sort=False)
    negative_df_with_md = pd.concat([train_neg_df_sp_and_t, test_neg_df_sp_and_t], ignore_index=True, sort=False)

    # Subsequent analysis
    seq_len_dist_plot(
        df_with_md,
        "SP",
        "Class",
        ["Negative", "Positive"],
        f"{config.config["data_analysis_dir"]}/seq_len_dist_by_class.pdf"
    )
    seq_len_dist_plot(
        df_with_md,
        "Training",
        "Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_dist_by_dataset.pdf"
    )
    seq_len_dist_plot(
        positive_df_with_md,
        "Training",
        "Positive Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_dist_by_dataset_pos.pdf"
    )
    seq_len_dist_plot(
        negative_df_with_md,
        "Training",
        "Negative Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_dist_by_dataset_neg.pdf"
    )

    seq_len_boxplot(
        df_with_md,
        "SP",
        "Class",
        ["Negative", "Positive"],
        f"{config.config["data_analysis_dir"]}/seq_len_boxplot_by_class.pdf"
    )
    seq_len_boxplot(
        df_with_md,
        "Training",
        "Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_boxplot_by_dataset.pdf"
    )
    seq_len_boxplot(
        positive_df_with_md,
        "Training",
        "Positive Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_boxplot_by_dataset_pos.pdf"
    )
    seq_len_boxplot(
        negative_df_with_md,
        "Training",
        "Negative Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/seq_len_boxplot_by_dataset_neg.pdf"
    )

    cleavage_len_boxplot(
        positive_df_with_md,
        "Training",
        "Positive Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/cleavage_len_boxplot_by_dataset_pos.pdf"
    )

    cleavage_len_his(
        positive_df_with_md,
        "Training",
        "Positive Datasets",
        ["Training", "Testing"],
        f"{config.config["data_analysis_dir"]}/cleavage_len_hist_by_dataset_pos.pdf"
    )

    # TODO Comparative amino-acid composition of SPs against some background distribution






