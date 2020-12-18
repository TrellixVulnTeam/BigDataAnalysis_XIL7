import numpy as np
import pandas as pd
import time
import re
import os
import pandas_profiling
import random
import seaborn as sns
import matplotlib.pyplot as plt


def generateMissingVal(df, ml):
    frac = ml
    idx = np.random.choice(range(df.shape[0]), int(df.count().sum() * frac), replace=True)
    cols = np.random.choice(range(df.shape[1]), size=len(idx), replace=True)
    x = df.astype(object).to_numpy()
    x[idx, cols] = np.nan
    df = pd.DataFrame(x, index=df.index, columns=df.columns)
    columns = df.columns
    c1 = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    columns = list(set(columns) - set(c1))
    for col in columns:
        df[col] = pd.to_numeric(df[col])

    return df


def calcIQR(df):
    num = df._get_numeric_data()
    Q1 = num.quantile(0.25)
    Q3 = num.quantile(0.75)
    IQR = Q3 - Q1
    lW = Q1 - 1.5 * IQR
    uW = Q3 + 1.5 * IQR
    return lW, uW


def showOutliners(col, df, uW, lW, kind):
    return df[(df[col] > uW[col]) | (df[col] < lW[col])]


def plotOutliners(col, df):
    sns.boxplot(data=df, y=col)
    df.boxplot(column=col)