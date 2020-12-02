import numpy as np
import pandas as pd

"""
Built-in functions:

df.style.highlight_max(color='darkorange', axis=None) # axis=None is max of whole dataframe
df.style.highlight_min(subset=['B'], axis=0) # axis=0 columnwise (default) and axis=1 for row wise
df.style.highlight_null('salmon') # null_color='red' is too bad
df.style.background_gradient(cmap='viridis',low=.5, high=0)
df.style.highlight_max(subset= pd.IndexSlice[1:3, ['B', 'D']]) # only max from given index range
df.style.bar(subset=['A', 'B'], align='mid', color=['#d65f5f', '#5fba7d'])
df.style.hide_columns(['C','D'])
df.style.hide_index()
df.style.set_caption('caption.')

"""

def highlight_row(ser, color="lightblue", row=None):
    if row is None:
        row = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == row else "" for _ in ser]

def highlight_col(ser, color="salmon", col=None):
    if col is None:
        col = ser.index[-1]
    bkg = f"background-color: {color}"
    return [bkg if ser.name == col else "" for _ in ser]

def highlight_diag(dfy, color="khaki"):
    a = np.full(dfy.shape, "", dtype="<U24")
    np.fill_diagonal(a, f"background-color: {color}")
    df1 = pd.DataFrame(a, index=dfy.index, columns=dfy.columns)
    return df1

def highlight_rowf(dfx, color="lightblue", row=None):
    if row is None:
        row = dfx.index[-1]

    def highlight_row(ser, row, color="salmon"):
        bkg = f"background-color: {color}"
        return [bkg if ser.name == row else "" for _ in ser]

    return dfx.style.apply(highlight_row, axis=1, row=row)


def highlight_colf(dfx, color="salmon", col=-1):
    if not isinstance(col, str):
        col = dfx.columns[col]

    def highlight_col(dfy):  # axis=None needs dataframe
        bkg = f"background-color: {color}"
        df1 = pd.DataFrame("", index=dfy.index, columns=dfy.columns)
        df1.loc[:, col] = bkg
        return df1

    return dfx.style.apply(highlight_col, axis=None)

def highlight_diagf(dfx, color="khaki"):
    def highlight_diag(dfy):
        a = np.full(dfy.shape, "", dtype="<U24")
        np.fill_diagonal(a, f"background-color: {color}")
        df1 = pd.DataFrame(a, index=dfy.index, columns=dfy.columns)
        return df1

    return dfx.style.apply(highlight_diag, axis=None)

def highlight_rcd(dfx, row=None, col=None, c1="lightblue", c2="salmon", c3="khaki"):
    return (
        dfx.style.apply(highlight_diag, axis=None, color=c3)
        .apply(highlight_row, axis=1, color=c1, row=row)
        .apply(highlight_col, axis=0, color=c2, col=col)
    )