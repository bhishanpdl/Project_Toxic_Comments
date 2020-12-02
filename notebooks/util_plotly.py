import numpy as np
import pandas as pd

import plotly
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff


# ===================== parameters ===========================
# Plot background color
paper_bgcolor = "rgb(240, 240, 240)"
plot_bgcolor = "rgb(240, 240, 240)"

# Red, blue, green (used by plotly by default)
rgb_def = ["rgb(228,26,28)", "rgb(77,175,74)", "rgb(55,126,184)"]

# Contrasting 2 qualities, highlighting one
contra_2_cols = ["rgb(150,150,150)", "rgb(55,126,184)"]

# Barchart axis templates
# template 1
bchart_xaxis_temp1 = dict(
    zeroline=False,
    showline=False,
    showgrid=False,
    showticklabels=False,
    tickfont=dict(size=9, color="grey"),
)

bchart_yaxis_temp1 = dict(tickfont=dict(size=9, color="grey"))

# template 2
bchart_xaxis_temp2 = dict(
    zeroline=False,
    showline=False,
    showgrid=False,
    showticklabels=False,
    tickfont=dict(size=10, color="grey"),
)

bchart_yaxis_temp2 = dict(tickfont=dict(size=10, color="grey"))

# Heatmap templates
heatmap_axis_temp1 = dict(
    zeroline=False, showline=False, showgrid=False, showticklabels=False, ticks=""
)


def plot_coo_matrix(labels, coo_matrix):
    fig_coords = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    axes_names = [
        ("x1", "y1"),
        ("x2", "y2"),
        ("x3", "y3"),
        ("x4", "y4"),
        ("x5", "y5"),
        ("x6", "y6"),
    ]
    axes_lo_names = [
        ("xaxis1", "yaxis1"),
        ("xaxis2", "yaxis2"),
        ("xaxis3", "yaxis3"),
        ("xaxis4", "yaxis4"),
        ("xaxis5", "yaxis5"),
        ("xaxis6", "yaxis6"),
    ]
    fig = make_subplots(
        rows=3,
        cols=2,
        horizontal_spacing=0.15,
        vertical_spacing=0.25,
        subplot_titles=(
            labels[0],
            labels[1],
            labels[2],
            labels[3],
            labels[4],
            labels[5],
        ),
    )
    for i, c_type, fig_coord, ax in zip(
        range(len(labels)), labels, fig_coords, axes_names
    ):
        inner_count = pd.Series(coo_matrix[i, :], index=labels)
        inner_count = inner_count.sort_values()
        trace = go.Bar(x=inner_count, y=list(inner_count.index), orientation="h")
        fig.append_trace(trace, fig_coord[0], fig_coord[1])

    fig["layout"].update(
        showlegend=False,
        title="<b>Horizontal Bar Plots</b>",
        xaxis1=bchart_xaxis_temp2,
        yaxis1=bchart_yaxis_temp2,
        xaxis2=bchart_xaxis_temp2,
        yaxis2=bchart_yaxis_temp2,
        xaxis3=bchart_xaxis_temp2,
        yaxis3=bchart_yaxis_temp2,
        xaxis4=bchart_xaxis_temp2,
        yaxis4=bchart_yaxis_temp2,
        xaxis5=bchart_xaxis_temp2,
        yaxis5=bchart_yaxis_temp2,
        xaxis6=bchart_xaxis_temp2,
        yaxis6=bchart_yaxis_temp2,
        margin=go.Margin(l=100, r=100, t=100, b=25,),
        autosize=False,
        width=900,
        height=900,
    )
    iplot(fig)

    # As a heatmap
    fig = ff.create_annotated_heatmap(
        z=coo_matrix,
        x=labels,
        y=labels,
        colorscale="YlGnBu",
        zmin=1,
        zmax=coo_matrix.max(),
    )
    fig["layout"]["xaxis"].update(side="bottom")
    fig["layout"].update(
        title="<b>Multilabel Confusion Matrix</b>",
        xaxis=dict(title="Predicted label (Vertical)", tickfont=dict(color="grey")),
        yaxis=dict(title="True label(Horizontal)", tickfont=dict(color="grey")),
        margin=go.Margin(l=150, r=150, t=150, b=75),
        autosize=False,
        width=900,
        height=450,
    )
    iplot(fig)
