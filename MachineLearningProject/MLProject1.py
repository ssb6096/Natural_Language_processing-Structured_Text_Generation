from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
df = pd.read_csv('all_ml_ideas.csv', header=None, names=['idea'])
df['idea'] = df['idea'].str.lower()
df.head()

valid_pct = 0.25 #validation percent
df = df.iloc[np.random.permutation(len(df))]
cut = int(valid_pct * len(df)) + 1
train_df, valid_df = df[cut:], df[:cut]
print(len(df))
print(len(train_df))
print(len(valid_df))
nan_rows = df[df['idea'].isnull()]
print(nan_rows)
#data_lm = TextLMDataBunch.from_csv('data','all_ml_ideas.csv')

data_lm = TextLMDataBunch.from_df('data', train_df, valid_df, text_cols='idea')

#print(data_lm)
learn = language_model_learner(train_df, pretrained_model=URLs.WT103, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)

wd=1e-7
lr=1e-3
lrs = lr
print(wd)
print(lr)
learn.fit(15,lrs, wd)