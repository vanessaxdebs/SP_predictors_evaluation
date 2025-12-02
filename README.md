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

 > Number of results (22/09/2025): **2,949**

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

 > Number of results (22/09/2025): **20,615**

#### b. Filtering the Positive Dataset
UniProtKB does not directly allow filtering signal peptides by length.

A custom Python script was implemented to:  
- Query UniProtKB via its REST API  
- Iterate through JSON results  
- Retain only proteins with signal peptides ≥ 14 residues  
- Export the final datasets in both `.tsv` and `.fasta` formats  
  
 > The final number of results in the positive dataset was: **2,938**
### Data Preparation

### Data Analysis 

### 

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

## Authors
This project was implemented thanks to an equal contribution of Nikita Leino and Vanessa El Debs