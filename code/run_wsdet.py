#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:42:56 2025

run Wittkuhn&Schuck method on the MEG data with various paradigms

@author: simon.kern
"""
import mne
from tqdm import tqdm
import pandas as pd
import bids_utils
import settings
from settings import layout
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import meg_utils
from meg_utils import sigproc
from meg_utils.plotting import savefig
from meg_utils.decoding import LogisticRegressionOvaNegX
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wsdet
