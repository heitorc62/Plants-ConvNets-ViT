import pandas as pd
import plotly.graph_objs as go
from functions import get_color

def compute_histogram(filtered_df):
    classes = filtered_df['manual_label'].value_counts().to_dict()

    map_colors = {}
    for index, row in filtered_df.iterrows():
        c = row['manual_label']
        if c not in map_colors:
            map_colors[c] = get_color(int(row['colors']))

    print()
    labels = [l + ' (' + str(c) + ')' for l, c in zip(classes.keys(), classes.values())]
    colors = [map_colors[c] for c in classes.keys()]
    total = 0
    for v in classes.values():
        total += v

    fig = go.Figure([go.Bar(x=labels, y=list(classes.values()), marker_color = colors)])
    fig.update_layout(
        title=str(total) + ' images selected.',
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig