# SP_predictors_evaluation

## Intro
This repository aims to compare two methods of SP prediction

## Steps
### Data Collection
**Objective:** retrieve positive and negative datasets of eukaryotic proteins from UniProtKB.

#### a. Selection criteria

**Positive dataset (secreted proteins with experimental SP evidence):**
  1. No fragments: Fragment: No (fragment:false)
  2. Select only eukaryotic proteins: Taxonomy [OC]: 2759 (taxonomy_id:2759)
  3. Filter-out sequences shorter than 40 residues: Sequence length: 40 to (length:[40 TO *])
  4. Filter-out unreviewed proteins: Reviewed: Yes (reviewed:true)
  5. Protein existence supported by experimental evidence: Protein Existance[PE]: Evidence at protein level (existence:1)
  6. Presence of experimentally validated signal peptide: Signal Peptide: * Evidence: Experimental (ft_signal_exp:*)
  7. Filter out proteins with SP shorter than 14 residues (custom filter, see below).

Final query: 
```bash
(existence:1) AND (length:[40 TO *]) AND (reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) AND (ft_signal_exp:*)
```

 > Number of results (15/11/2025): **2,938**

**Negative dataset (Non-secretory proteins, with defined subcellular localization):**
  1. No fragments: Fragment: No (fragment:false)
  2. Only reviewed proteins: Reviewed: Yes (reviewed:true)
  3. Select on protein with experimental SP evidence: Protein Existance[PE]: Evidence at protein level (existence:1)
  4. Only eukaryotic proteins: Taxonomy [OC]: 2759 (taxonomy_id:2759)
  5. Sequence length ≥ 40 residues: Sequence length: 40 to (length:[40 TO *])
  6. Exclude all proteins with any signal peptide: Signal Peptide: * Evidence: Any (NOT ft_signal_exp:*)
  7. Experimentally localized to cytosol, nucleus, mitochondrion, plastid, peroxisome, or cell membrane: Subcellular location term: SL-0173/SL-0191/SL-0173/SL-0204/SL-0209/SL-0039 Evidence: Experimental

Final query: 
```bash
(existence:1) AND (length:[40 TO *]) AND (reviewed:true) AND (fragment:false) AND (taxonomy_id:2759) NOT (ft_signal:*) AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0039))
```

 > Number of results (15/11/2025): **20,615**

#### b. Filtering the Positive Dataset
UniProtKB does not directly allow filtering signal peptides by length.

A custom Python script was implemented to:  
- Query UniProtKB via its REST API  
- Iterate through JSON results  
- Retain only proteins with signal peptides ≥ 14 residues  
- Export the final datasets in both `.tsv` and `.fasta` formats  
  
 > The final number of results in the positive dataset was: **2,938**
### Data Preparation
**Objective:** remove sequence redundancy and split the curated UniProtKB datasets into training, validation, and test sets.

This step is implemented in `scripts/data_preparation.py` and operates on the `.tsv`/`.fasta` files produced in the *Data Collection* step.

- **Clustering and redundancy reduction (MMseqs2):**
  - For both positive and negative FASTA files, `mmseqs easy-cluster` is called (via `exec_mmseqs_easy_cluster`) with:
    - **Minimum sequence identity:** `config["min_seq_id"]`
    - **Coverage threshold:** `config["coverage_threshold"]`
  - MMseqs2 groups similar proteins into clusters and selects a **representative sequence** per cluster.
  - Output is written under `data/data_preparation/<prefix>/`, where `<prefix>` is `positive_prefix` or `negative_prefix`.

- **Selection of representative sequences:**
  - `filter_redundant_seq` reads:
    - `cluster-results_cluster.tsv` (cluster ↔ member mapping from MMseqs2),
    - the original `positive.tsv` or `negative.tsv` from *Data Collection*.
  - It keeps only the rows whose **ID** is a cluster representative.
  - The resulting non‑redundant tables are saved as:
    - `positive_filtered.tsv`, `negative_filtered.tsv` in `data/data_preparation/`.

- **Train / test / validation splitting:**
  - `train_test_validation_split` splits each non‑redundant dataset as follows:
    - **80%** of sequences → training set,
    - **20%** of sequences → test set.
  - A `Group` column is added to the training set:
    - `Group = 100` initially, then reassigned so that the training data is split into **5 equal folds**:
      - Folds are indexed from `1` to `5` and are used for downstream **cross‑validation**.
  - The following files are produced for each class (positive/negative):
    - `<prefix>_train.tsv` — full training set with `Group` labels,
    - `<prefix>_test.tsv` — held‑out test set,
    - `<prefix>_train_0.tsv` … `<prefix>_train_4.tsv` — per‑fold validation subsets.

These files are used as the main input for all subsequent steps (exploratory data analysis, PSWM, and SVM).

### Data Analysis 
**Objective:** perform exploratory analysis of the curated datasets to check for biases and to characterize the protein and SP length distributions.

This step is implemented in `scripts/data_analysis.py` and reads the train/test splits produced in *Data Preparation*:
- `positive_train.tsv`, `negative_train.tsv`
- `positive_test.tsv`, `negative_test.tsv`

The script combines these tables and generates a series of **PDF plots** in `data/data_analysis/`:

- **Kingdom distributions:**
  - `dist_by_kingdom_plot` creates side‑by‑side **pie** and **bar** plots of the `Kingdom` column.
  - Generated for:
    - Whole training set (`train_kingdoms.pdf`),
    - Positive/negative training subsets (`pos_train_kingdoms.pdf`, `neg_train_kingdoms.pdf`),
    - Whole test set and its positive/negative subsets (`test_kingdoms.pdf`, `pos_test_kingdoms.pdf`, `neg_test_kingdoms.pdf`).

- **Species distributions:**
  - `dist_by_species_plot` collapses rare species into **“Other”** and plots the most frequent taxa.
  - Figures are produced for train/test and per‑class subsets (e.g. `train_species.pdf`, `pos_train_species.pdf`, etc.).

- **Protein sequence length distributions:**
  - `seq_len_dist_plot` uses kernel density estimation on the `Seq_Length` column to compare:
    - Positive vs negative classes (`seq_len_dist_by_class.pdf`),
    - Training vs test sets (`seq_len_dist_by_dataset.pdf`),
    - Training vs test within positive or negative classes (`seq_len_dist_by_dataset_pos.pdf`, `seq_len_dist_by_dataset_neg.pdf`).
  - `seq_len_boxplot` shows the same comparisons using boxplots (`seq_len_boxplot_by_*` files).

- **Signal peptide length distributions (positives only):**
  - After tagging sequences with `SP` (0/1) and `Training` (0/1), the script focuses on the `Site` column (cleavage position).
  - `cleavage_len_boxplot` and `cleavage_len_his` compare SP length in training vs test for positives:
    - `cleavage_len_boxplot_by_dataset_pos.pdf`
    - `cleavage_len_hist_by_dataset_pos.pdf`

Overall, this step validates that training and test sets have compatible distributions and that no major taxonomic or length biases are introduced.

### Training Set Preparation
**Objective:** build PSWM‑ready training and test tables, including sequence fragments around the cleavage site and N‑terminal fragments of fixed length.

Implemented in `scripts/pswm_training_set_preparation.py`, this step bridges the *Data Preparation* output and the PSWM and SVM models.

- **Input tables:**
  - `positive_train.tsv`, `negative_train.tsv`
  - `positive_test.tsv`, `negative_test.tsv`

- **Class labelling and feature selection:**
  - Adds a `Class` column:
    - `1` for positives, `0` for negatives.
  - Keeps only the columns necessary for modelling:
    - `ID`, `Seq_Length`, `Site`, `Class`, `Group`.

- **Merging FASTA files:**
  - `merge_fasta_files` concatenates:
    - `negative.fasta` and `positive.fasta` (from `data/data_collection/`)
  - into a single `train.fasta` stored in `data/pswm/`.

- **Adding full sequences and derived fragments:**
  - `add_sequences_from_fasta` maps UniProt IDs (`ID`) to amino‑acid sequences from `train.fasta`.
  - `add_fragment_columns` creates:
    - `Frag_90`: first 90 residues of the protein (N‑terminal fragment used for scanning),
    - `SP_15`: a 15‑residue window centred on the cleavage site:
      - from position `Site − 13` to `Site + 2` (1‑based, inclusive),
      - or `"None"` when this window cannot be extracted.

- **Final PSWM‑ready tables:**
  - Training: `train.tsv` in `data/pswm/` (used to compute the PSWM and to train the SVM features).
  - Testing: `test.tsv` in `data/pswm/` (used for final evaluation of both methods).

### von-Heijne method
**Objective:** implement a PSWM‑based signal peptide predictor inspired by von Heijne, train it on curated SP windows, and evaluate it with cross‑validation and on an independent test set.

This step is implemented in `scripts/pswm.py` and relies on `train.tsv` and `test.tsv` generated in *Training Set Preparation*.

- **PSWM construction:**
  - `compute_matrix` builds a **Position‑Specific Weight Matrix** from the `SP_15` column:
    - Uses only **positive SP windows** (`Class = 1`) from selected training folds (e.g. Groups 1–3).
    - Starts from pseudo‑counts (Laplace smoothing) and normalizes by background SwissProt amino‑acid frequencies.
    - Converts probabilities to log\$_2\$ scores, resulting in an enrichment matrix for each residue/position.
  - The resulting matrix is stored as `matrix.npz` (and `matrix_cv_*.npz` for cross‑validation runs) in `data/pswm/`.

- **Scoring sequences with the PSWM:**
  - `compute_score` slides a 15‑residue window across `Frag_90` (N‑terminal fragment) and:
    - sums the PSWM scores at each position,
    - returns the **maximum** window score as the sequence score.

- **Threshold selection (internal validation):**
  - For a given training/validation split, scores are computed on a validation fold (e.g. `Group = 4`).
  - `calc_threshold`:
    - computes the **precision–recall curve**,
    - finds the score threshold that maximizes the **F1‑score**,
    - plots the curve and optimal point to `threshold.pdf` (or `threshold_cv_*.pdf`).

- **Prediction and metrics on held‑out groups:**
  - `predict` applies the selected threshold to assign `Prediction` ∈ {0,1} to each sequence.
  - The script writes:
    - `prediction.tsv` (predictions on a held‑out training group),
    - `metrics.tsv` (Accuracy, MCC, Recall, Precision, F1 on that group).

- **Five‑fold cross‑validation:**
  - For each `Group` (1–5), the method:
    - trains a new PSWM on the remaining four folds (positives only),
    - tunes a threshold on the validation fold,
    - evaluates performance and logs metrics.
  - The per‑fold metrics are collected in `cv_metrics.tsv`, and PSWM heatmaps / threshold curves are stored as:
    - `PSWM_cv_*.pdf`, `threshold_cv_*.pdf`.

- **Final testing:**
  - The PSWM and threshold from the **best F1** cross‑validation iteration are selected.
  - These are applied to `test.tsv`, and the results are saved as:
    - `test_df_w_prediction.tsv` — predictions with full metadata,
    - `test_df_metrics.tsv` — final metrics on the independent test set,
    - `confusion_matrix.pdf` — confusion matrix summary.

This step represents the von‑Heijne‑like method that you later compare to the SVM‑based predictor.

### SVM
**Objective:** train a supervised SVM classifier on physically interpretable sequence‑derived features and compare its performance to the PSWM/von‑Heijne approach.

The SVM pipeline is implemented in `scripts/svm.py` and uses `train.tsv` and `test.tsv` from the *Training Set Preparation* step.

- **Feature engineering from N‑terminal fragments:**
  - All SVM features are computed from `Frag_90` (N‑terminal fragment) and are biologically motivated:
    - **Amino‑acid composition** of the first 40 residues (`sequence_composition`).
    - **Hydrophobicity and aliphatic index profiles** (`hp_ai`), derived from the Kyte–Doolittle scale and ProParam‑like measures.
    - **Secondary structure propensities** (`SSE`): weighted helix and sheet scores over sliding windows.
    - **Charge distribution** in the first 20 residues (`charge_seq`) using simple residue charges.
    - **Transmembrane helix propensity** (`tm_helix_propensity`), capturing how “TM‑like” the N terminus is.
  - `extract_features` concatenates all these components into a 39‑dimensional feature vector per sequence.

- **Baseline SVM and hyperparameter search:**
  - `svm_pipeline` wraps a `StandardScaler` and an RBF‑kernel `SVC`.
  - For each cross‑validation round, the script:
    - splits `train.tsv` into training and validation folds based on the `Group` column,
    - extracts features for train/validation/test sequences,
    - performs a grid search over `C` and `gamma` on the validation fold using **MCC** as the objective.

- **Random Forest feature importance and feature selection:**
  - A `RandomForestClassifier` is trained on the full feature set to compute **Gini importances**.
  - Importances are saved as `Features_gini_<round>.tsv`, and the top 20 features are visualized as:
    - `Top_Features_round<round>.pdf` and `top_20_features_<round>.pdf`.
  - The script then evaluates SVM MCC as a function of the number of top‑Gini features (k) using `mcc_subset`:
    - Curves are plotted as `MCC_vs_Val_round<round>.pdf` and `top_20_features_RF_gini_<round>.pdf`.
  - The best-performing `k` and corresponding feature subset are tracked across the 5 rounds.

- **Final model training and evaluation:**
  - After summarizing the best feature subset over CV, the script:
    - re‑extracts features for a final train/validation split,
    - re‑optimizes SVM hyperparameters on the selected features,
    - trains a **final SVM model** using the selected features and best hyperparameters.
  - The model is stored as `model.pkl.gz` in `data/svm/`.
  - Predictions on the independent test set are saved as:
    - `test_df_w_prediction.tsv`, including the `Prediction` column,
    - Confusion matrix plots for:
      - **Selected‑feature model** (`confusion_matrix_Selected features.pdf`),
      - **All‑features baseline** (`confusion_matrix_All Features.pdf`).
  - Summary metrics (MCC, precision, recall, accuracy, F1) are pickled for:
    - Selected‑feature model: `selected_metrics_mcc.pkl.gz`,
    - All‑features baseline: `all_metrics_mcc.pkl.gz`.

This step yields the SVM‑based predictor that is compared to the von‑Heijne PSWM method.

### Results Analysis
**Objective:** perform a comparative, mechanistic analysis of the PSWM and SVM predictions, with a focus on transmembrane (TM) proteins and SP sequence motifs.

This step is implemented in `scripts/results_analysis.py` and uses:
- `svm/test_df_w_prediction.tsv` — SVM predictions on the test set,
- `pswm/test_df_w_prediction.tsv` — PSWM predictions on the same test set,
- `pswm/train.tsv` — PSWM training table (for background logos),
- `negative.tsv` from `data/data_collection/` — negatives annotated with a TM flag.

The script focuses on two main analyses:

- **Fraction of TM proteins among false positives:**
  - `extract_fp_df` and `assign_tm` identify **false positives (FP)** (Prediction = 1, Class = 0) for both methods.
  - `get_tm_fraction` merges FPs with the negative dataset to retrieve the `TM` boolean flag.
  - It then computes the fraction of FP sequences that contain a **predicted or annotated transmembrane helix**:
    - This allows assessing whether each method tends to confuse TM helices with signal peptides.

- **Sequence logos of PSWM false negatives vs. training positives:**
  - `extract_fn_df` extracts **false negatives (FN)** from PSWM predictions (Prediction = 0, Class = 1) along with their `SP_15` windows.
  - `plot_sequence_logo` generates sequence logos using `logomaker` for:
    - `fn_sp_15_sequence_logo.pdf` — SP_15 windows of FN predictions (patterns that PSWM systematically misses),
    - `train_sp_15_sequence_logo.pdf` — SP_15 windows from the positive PSWM training set (what the model was trained on).
  - Comparing these logos helps reveal **motif differences** between “easy” and “hard” signal peptides for the von‑Heijne method.

Together, these analyses provide an overview of possible error reasons.

## Installation
### Conda
1.Clone the repository
```shell
git clone
```
2. Create an environment
```shell
conda env create -f SP_pred_eval.yml
```
3. Activate the environment
```shell
conda activate SP_pred_eval
```

## Execution
Before execution check config file (for now it is ./config/config.py) and change the variables
the way you prefer.

The whole project can be done in one command:
```shell
python3 main.py
```

However, there is also a possibility to execute step by step: (DON'T GO INSIDE DIRECTORIES, EXECUTE FROM ROOT)
### Data Collection
```shell
python3 -m scripts.data_collection
```
### Data Preparation
```shell
python3 -m scripts.data_preparation
```
### Data Analysis
```shell
python3 -m scripts.data_analysis
```
### Training set preparation
```shell
python3 -m scripts.pswm_training_set_preparation
```
### von-Heijne method 
```shell
python3 -m scripts.pswm
```
### SVM (Support Vector Machine)
```shell
python3 -m scripts.svm
```
### Analysis of the results
```shell
python3 -m scripts.results_analysis
```

## Authors
This project was implemented thanks to an equal contribution of Nikita Leino and Vanessa El Debs