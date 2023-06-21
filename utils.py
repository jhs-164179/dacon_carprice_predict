import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# scatter plot 확인
def plot_scatter_with_fixed_col(df, fixed_col, hue=False, drop_cols=[], size=5, title='', n_cols=3):    
    if hue:
        cols = df.columns.drop([hue, fixed_col] + drop_cols)
    else:
        cols = df.columns.drop([fixed_col] + drop_cols)
    n_cols = n_cols
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows), sharex=False, sharey=False)
    fig.suptitle(f'{title} Set Scatter Plot with Target Column by {hue}', fontsize=20, fontweight='bold', y=1)

    for i, col in enumerate(cols):
        n_row = i // n_cols
        n_col = i % n_cols
        ax = axes[n_row, n_col]

        ax.set_xlabel(f'{col}', fontsize=12)
        ax.set_ylabel(f'{fixed_col}', fontsize=12)

        # Plot the scatterplot
        if hue:
            sns.scatterplot(data=df, x=col, y=fixed_col, hue=hue, ax=ax,
                            s=40, edgecolor='gray', alpha=1, palette='bright')
            ax.legend(title=hue, title_fontsize=12, fontsize=12) # loc='upper right'
        else:
            sns.scatterplot(data=df, x=col, y=fixed_col, ax=ax,
                            s=40, edgecolor='gray', alpha=1)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_title(f'{col}', fontsize=16)
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()

# 이상치 관리 함수
def manage_outlier(df, col, how='capping', target='normal'):
    Q1 = np.quantile(df[col], .25)
    Q3 = np.quantile(df[col], .75)
    IQR = Q3 - Q1
    if target=='normal':
        maximum = Q3 + (1.5 * IQR)
        minimum = Q1 - (1.5 * IQR)
    if target=='extreme':
        maximum = Q3 + (3 * IQR)
        minimum = Q1 - (3 * IQR)

    if how=='capping':
        df.loc[df[col]>maximum, col] = maximum
        df.loc[df[col]<minimum, col] = minimum
    if how=='delete':
        del_idx = df[np.logical_or(df[col]>maximum, df[col]<minimum)].index
        df.drop(index=del_idx, inplace=True)
        df.reset_index(inplace=True)
        df.drop('index',axis=1,inplace=True)