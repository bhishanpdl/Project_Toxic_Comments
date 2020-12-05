import numpy as np
import pandas as pd

def get_df_coo(y_true,y_pred,column_names):
    """
    Get Co-occurence matrix from test labels and predictions.
    """
    yt = np.array(y_true,dtype=np.int32)
    yp = np.array(y_pred,dtype=np.int32)
    coo = yt.T.dot(yp)
    df_coo = pd.DataFrame(coo, columns=column_names,index=column_names)
    df_coo.loc['Total']= df_coo.sum(numeric_only=True, axis=0)
    df_coo.loc[:,'Total'] = df_coo.sum(numeric_only=True, axis=1)
    df_coo = df_coo.astype(np.int32)
    return df_coo

def highlight_diagf(dfx, color="khaki"):
    def highlight_diag(dfy):
        a = np.full(dfy.shape, "", dtype="<U24")
        np.fill_diagonal(a, f"background-color: {color}")
        df1 = pd.DataFrame(a, index=dfy.index, columns=dfy.columns)
        return df1

    return dfx.style.apply(highlight_diag, axis=None)