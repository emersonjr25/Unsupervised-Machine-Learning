"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
UNSUPERVISIONED MACHINE LEARNING
"""

import pandas as pd

data_general = pd.read_csv('data/Assignment-1_Data.csv', sep=';')

data_general.isna().sum()

data_general = data_general.dropna()



