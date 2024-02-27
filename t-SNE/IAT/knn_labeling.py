from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
import sys

def give_labels(ind, colors):
    Y_colors = []
    for i in range(ind.shape[0]):
        count_colors = np.zeros(colors.max()+1)
        closest = ind[i]
        for c in closest:
            count_colors[colors[c]] += 1
        Y_colors.append(np.argmax(count_colors))
        
    return Y_colors

def run_knn(labeled_csv, unlabeled_csv, k = 5):
    labeled_df = pd.read_csv(labeled_csv)
    unlabeled_df = pd.read_csv(unlabeled_csv)
    output_path = unlabeled_csv

    X = labeled_df[['x', 'y']]
    Y = unlabeled_df[['x', 'y']]
    labels = labeled_df['manual_label']
    colors = labeled_df['colors']
    label_2_color = {}
    color_2_label = {}

    for i in range(labels.size):
        label = labels[i]
        color = colors[i]
        label_2_color[label] = color
        color_2_label[color] = label

    colors = np.array([label_2_color[x] for x in labels])

    kdt = KDTree(X, leaf_size=30, metric='euclidean')

    dist, ind = kdt.query(Y, k=k)     
    Y_colors = give_labels(ind, colors)
    Y_labels = [color_2_label[x] for x in Y_colors]

    unlabeled_df['colors'] = Y_colors
    unlabeled_df['manual_label'] = Y_labels

    labeled_df['colors'] = [label_2_color[x] for x in labels]
    labeled_df['manual_label'] = labels

    unlabeled_df.to_csv(output_path, index=False)

    return output_path
