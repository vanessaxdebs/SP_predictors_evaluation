import pandas as pd
import numpy as np
import pickle, gzip
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, PrecisionRecallDisplay, accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import math

def sequence_composition(sequence:str) -> np.ndarray[np.float64]:
  sequence = sequence[:40]
  alphabet = "AQLSREKTNGMWDHFYCIPV"
  composition_matrix = np.zeros((20,40))
  composition_vector = np.zeros((20))
  for i in range(len(alphabet)):
    for j in range(len(sequence)):
      if alphabet[i] == sequence[j]:
        composition_matrix[i][j] += 1
        composition_vector[i] += composition_matrix[i][j]
  composition_vector = composition_vector/40
  return composition_vector

def hp_ai(sequence:str, window_size:int = 6 ) -> np.ndarray[np.float64]:
  kd_scale = {
    'R': -4.5,  # Arg
    'K': -3.9,  # Lys
    'N': -3.5,  # Asn
    'D': -3.5,  # Asp
    'Q': -3.5,  # Gln
    'E': -3.5,  # Glu
    'H': -3.2,  # His
    'P': -1.6,  # Pro
    'Y': -1.3,  # Tyr
    'W': -0.9,  # Trp
    'S': -0.8,  # Ser
    'T': -0.7,  # Thr
    'G': -0.4,  # Gly
    'A':  1.8,  # Ala
    'M':  1.9,  # Met
    'C':  2.5,  # Cys
    'F':  2.8,  # Phe
    'L':  3.8,  # Leu
    'V':  4.2,  # Val
    'I':  4.5   # Ile
    }

  a = 2.9 # relative volume of valine
  b = 3.9 # relative volume leucine/isoleucine

  sequence = sequence[:40]
  # padding to have as many scores as residues  (each window refers to the central res)
  d = int(window_size/2)
  sequence = "X"*d + sequence + "X"*d #padding
  hydrophobicities = np.array([])
  AIs = np.array([])  #aliphatic indexes
  for i in range(len(sequence)-(window_size)+1):
      counts_A = 0
      counts_V = 0
      counts_I = 0
      counts_L = 0
      hydrophobicity_score = 0
      window = sequence[i:i+window_size]
      for j in range(window_size):
        if window[j] != "X":
          hydrophobicity_score = (hydrophobicity_score + kd_scale[window[j]])
          if window[j] == 'A':
              counts_A += 1
          elif window[j] == 'V':
              counts_V += 1
          elif window[j] == 'I':
              counts_I += 1
          elif window[j] == 'L':
                counts_L += 1
      X_A = (counts_A / window_size)
      X_V = (counts_V / window_size)
      X_I = (counts_I / window_size)
      X_L = (counts_L / window_size)
      AI = X_A + a * X_V + b * (X_I + X_L) #Aliphatic Index Formula
      hydrophobicities = np.append(hydrophobicities, hydrophobicity_score / window_size)
      AIs = np.append(AIs, AI)
  H_AI = np.array([hydrophobicities.mean(), hydrophobicities.max(), np.argmax(hydrophobicities), AIs.mean(), AIs.max(), np.argmax(AIs)])
  return H_AI  #hydrophobicity and aliphatic index alltogether

def SSE(sequence,window_size):
  alpha_helix_scale = {
      "G": 0.570,  # Gly
      "P": 0.570,  # Pro
      "Y": 0.690,  # Tyr
      "C": 0.700,  # Cys
      "S": 0.770,  # Ser
      "T": 0.830,  # Thr
      "N": 0.670,  # Asn
      "R": 0.980,  # Arg
      "H": 1.000,  # His
      "D": 1.010,  # Asp
      "I": 1.080,  # Ile
      "W": 1.080,  # Trp
      "Q": 1.110,  # Gln
      "F": 1.130,  # Phe
      "K": 1.160,  # Lys
      "V": 1.060,  # Val
      "L": 1.210,  # Leu
      "A": 1.420,  # Ala
      "M": 1.450,  # Met
      "E": 1.510   # Glu
  }
  beta_sheet_scale = {
      "E": 0.370,  # Glu
      "D": 0.540,  # Asp
      "P": 0.550,  # Pro
      "G": 0.750,  # Gly
      "S": 0.750,  # Ser
      "K": 0.740,  # Lys
      "H": 0.870,  # His
      "N": 0.890,  # Asn
      "R": 0.930,  # Arg
      "A": 0.830,  # Ala
      "M": 1.050,  # Met
      "Q": 1.100,  # Gln
      "C": 1.190,  # Cys
      "T": 1.190,  # Thr
      "L": 1.300,  # Leu
      "F": 1.380,  # Phe
      "W": 1.370,  # Trp
      "Y": 1.470,  # Tyr
      "I": 1.600,  # Ile
      "V": 1.700   # Val
  }
  sequence = sequence[:40]
  # padding to have as many scores as residues (each window refers to the central res)
  d = int(window_size/2)
  sequence = "X"*d + sequence + "X"*d #padding
  alpha_helix = np.array([])
  beta_sheet = np.array([])
  for i in range(len(sequence)-(window_size)+1):
      alpha_score = 0
      beta_score = 0
      window = sequence[i:i+window_size]
      for j in range(window_size):
        w = 1 - abs(j - (window_size - 1)/2) / ((window_size - 1)/2)
        if window[j] != "X":
          alpha_score = alpha_score + w * alpha_helix_scale[window[j]]
          beta_score = beta_score + w * beta_sheet_scale[window[j]]
      alpha_helix = np.append(alpha_helix, alpha_score / window_size)
      beta_sheet = np.append(beta_sheet, beta_score / window_size)
  alpha_feature = np.array([alpha_helix.mean(), alpha_helix.max(), np.argmax(alpha_helix)])
  beta_feature = np.array([beta_sheet.mean(), beta_sheet.max(), np.argmax(beta_sheet)])
  return alpha_helix, alpha_feature, beta_sheet, beta_feature

def charge_seq(sequence, window_size):
  res_charges = {
        'K': 1,   # Lys
        'R': 1,   # Arg
        'H': 0.5, # His (partial positive)
        'D': -1,  # Asp
        'E': -1   # Glu
    }

  sequence = sequence[:20]
  # padding to have as many scores as residues  (each window refers to the central res)
  d = int(window_size/2)
  sequence = "X"*d + sequence + "X"*d #padding

  norm_charges = np.array([])

  for i in range(len(sequence)-(window_size)+1):
      charge = 0
      window = sequence[i:i+window_size]
      for j in range(window_size):
        if window[j] != "X" and window[j] in res_charges:
          charge = (charge + res_charges[window[j]])
      norm_charges = np.append(norm_charges, charge / window_size)
  charge_feature = np.array([norm_charges.max(), np.argmax(norm_charges), norm_charges.min(), np.argmin(norm_charges)])
  return norm_charges, charge_feature


def tm_helix_propensity(sequence: str, window_size: int = 7) -> np.ndarray[np.float64]:
    #TM Helix Propensity Scale (ProParam)
    tm_helix_propensity_scale = {
        'F':  1.980,
        'I':  1.970,
        'L':  1.820,
        'W':  1.530,
        'V':  1.460,
        'M':  1.400,
        'Y':  0.490,
        'A':  0.380,
        'G': -0.190,
        'C': -0.300,
        'T': -0.320,
        'S': -0.530,
        'H': -1.440,
        'P': -1.440,
        'N': -1.620,
        'Q': -1.840,
        'R': -2.570,
        'E': -2.900,
        'D': -3.270,
        'K': -3.460
    }

    sequence = sequence[:80]

    d = int(window_size / 2)     #Padding
    sequence = "X" * d + sequence + "X" * d

    tm_scores = np.array([])

    for i in range(len(sequence) - window_size + 1):
        score = 0.0
        window = sequence[i:i + window_size]

        for j in range(window_size):   #Weighted average
            w = 1 - abs(j - (window_size - 1) / 2) / ((window_size - 1) / 2)
            if window[j] in tm_helix_propensity_scale:
                score += w * tm_helix_propensity_scale[window[j]]
        tm_scores = np.append(tm_scores, score / window_size)

    tm_feature = np.array([
        tm_scores.mean(),        #mean score across sequence
        tm_scores.max(),         #max score
        np.argmax(tm_scores)     #position of max
    ])

    return tm_scores, tm_feature

def extract_features(seqs):
  features = np.empty([len(seqs), 39])

  for row, seq in enumerate(seqs):
    feature = np.array(())
    aa_comp = sequence_composition(seq)
    hyd_ali_index = hp_ai(seq)
    _, alpha, _, beta = SSE(seq, window_size=9)
    _, charges = charge_seq(seq, window_size=3)
    _, tm_propensity = tm_helix_propensity(seq, window_size=7)

    feature = np.concatenate((aa_comp, hyd_ali_index, tm_propensity, alpha, beta, charges), axis = None)

    features[row] = feature
  # Return individual feature vectors
  return features

if __name__ == "__main__":
    print("test")