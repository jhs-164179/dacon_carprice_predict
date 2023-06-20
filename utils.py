import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns

# 왜도, 첨도 확인
def univariate_stats(df):
    
    output_df = pd.DataFrame(columns=['Skew', 'Kurt'])
    
    for col in df:
        if is_numeric_dtype(df[col]):
            output_df.loc[col] = [df[col].skew(), df[col].kurt() ]
        else:
            output_df.loc[col] = ['-', '-' ]

    return output_df.sort_values(by=['Skew', 'Kurt'], ascending=True)

# 히스토그램 확인
def plot_hist(df, title, numeric_cols):
    
    cols = numeric_cols
    n_cols = 3
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8,6))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.histplot(df, x=var_name, kde=True, ax=ax)
        # sns.distplot(df[var_name], kde=True, ax=ax)
        ax.set_title(f'{var_name}')

    fig.suptitle(f'{title}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

# boxplot 확인
def plot_boxplot(df, hue=False, hues='', title='', drop_cols=[], n_cols=3):
    if hue:
        cols = df.columns.drop([hues] + drop_cols)
    else:
        cols = df.columns.drop(drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    if hue:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 6))
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 6))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        if hue:
            sns.boxplot(data=df, x=hues, y=var_name, ax=ax, showmeans=True, 
                        meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
            ax.set_title(f'{var_name} by {hues}')
            ax.set_xlabel('')
        else:
            sns.boxplot(data=df[var_name], ax=ax, showmeans=True, 
                        meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
            ax.set_title(f'{var_name}')
            ax.set_xlabel('')
    if hue:
        fig.suptitle(f'{title} Boxplot by {hues}', fontweight='bold', fontsize=16)
    else:
        fig.suptitle(f'{title} Boxplot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

# violinplot 확인
def plot_violinplot(df, hue=False, hues='', title='', drop_cols=[], n_cols=2):
    if hue:
        cols = df.columns.drop([hues] + drop_cols)
    else:
        cols = df.columns.drop(drop_cols)
    n_rows = (len(cols) - 1) // n_cols + 1

    if hue:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 6))
    else:
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 6))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        if hue:
            sns.violinplot(data=df, x=hues, y=var_name, ax=ax, inner='quartile')
        else:
            sns.violinplot(data=df[var_name], ax=ax, inner='quartile')
        ax.set_title(f'{var_name} Distribution')
    if hue:
        fig.suptitle(f'{title} Violin Plot by {hues}', fontweight='bold', fontsize=16)
    else:
        fig.suptitle(f'{title} Violin Plot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()