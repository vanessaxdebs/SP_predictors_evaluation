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

def compute_score (sequence,  window, pswm, alphabet = "AQLSREKTNGMWDHFYCIPV"):
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

if __name__ == "__main__":
    # Set theme in Seaborn
    sns.set_theme()
    mpl.rcParams['font.family'] = ['Liberation Serif', 'serif']
    mpl.rcParams['font.serif'] = ['Liberation Serif']
    mpl.rcParams['axes.unicode_minus'] = False  # avoid minus rendering issues
    sns.set_theme(context='notebook', style='white', palette='viridis', font='Liberation Serif', font_scale=1.1)

    df = pd.read_csv(f"{config.config['pswm_dir']}/train.tsv", sep = "\t")
    print(df.tail())

    # Compute matrix
    group_mask = df['Group'].isin([1,2,3])
    class_mask = df['Class'] == 1
    combined_mask = group_mask & class_mask
    filtered_df = df[combined_mask].copy()
    pwsm = compute_matrix(filtered_df['SP_15'], f"{config.config['pswm_dir']}/matrix")

    # compute score
    sequences = df[df['Group'] == 4]
    sequences = sequences['Frag_90']
    scores = np.array([])
    for seq in sequences:
        score = compute_score(seq, 15, pwsm)
        scores = np.append(scores, score)
    print((len(scores)))
