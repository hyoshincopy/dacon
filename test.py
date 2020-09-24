from IPython.display import clear_output
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm



from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
PATH = './train/'
with open('train.csv', 'r') as csv_file:
    data = csv_file.read()

all_captions = []
all_img_name_vector = []

for line in data.split('\n')[1:-1]:
    image_id, smiles = line.split(',')
    caption = '<' + smiles + '>'
    full_image_path = PATH + image_id

    all_img_name_vector.append(full_image_path)
    all_captions.append(caption)

train_captions, img_name_vector = shuffle(
    all_captions, all_img_name_vector, random_state=42)

num_examples = 908765  # 학습에 사용할 데이터 수, Baseline에서는 제공된 데이터 모두 사용하였습니다.
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
