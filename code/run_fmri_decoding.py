# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 08:48:42 2025

@author: Simon.Kern
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import settings
import numpy as np

# ---- load data -----
df = pd.read_csv('C:/Users/simon.kern/Desktop/highspeed-decoding/decoding/sub-01/data/sub-01_decoding.csv')   # change path / ID as needed

df_odd = df[(df.test_set=='test-odd_long') & (df.classifier=='log_reg')]

df_odd['target'] = df_odd['class']==df_odd['stim']
df_odd['accuracy'] = df_odd['pred_label']==df_odd['stim']

plt.rcParams.update({'font.size':14})
fig, ax = plt.subplots(1, 1, figsize=[4, 3])
sns.lineplot(df_odd, x='seq_tr', y='accuracy', ax=ax)
ax.hlines(0.2, 1, 7, color='black', alpha=0.5, linestyle='--')
ax.set_ylim(0, 0.8)
ax.set_xticks(np.arange(1, 8), [f'TR {x}\n{x*1.25:.2f}s' for x in range (1, 8)])
ax.legend(['decoding acc.', 'SE', 'chance level'], fontsize=10, loc='upper left')
ax.set_xlabel('timepoint after stim onset')
sns.despine()
plt.pause(0.1)
plt.tight_layout()

# # ---- keep oddball predictions from the multinomial model --------------
# odd = df[df['test_set'].isin(['test-odd_peak', 'test-odd_long'])]
# odd = odd[odd['classifier'] == 'log_reg']

# # ---- average probability across trials --------------------------------
# mean_p = (df_odd
#           .groupby(['seq_tr', 'class', 'stim'])['probability']
#           .mean()
#           .reset_index()
#           .pivot(index=('seq_tr'), columns='class', values='probability'))

# # ---- plot --------------------------------------------------------------
# fig, ax = plt.subplots()
# mean_p.plot(ax=ax, marker='o')
# ax.set_xlabel('TR within trial window (seq_tr)')
# ax.set_ylabel('Mean predicted probability')
# ax.set_title('Oddball decoding – multinomial logistic model')
# # ax.set_ylim(0, 1)
# ax.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
# plt.tight_layout()
# plt.show()
