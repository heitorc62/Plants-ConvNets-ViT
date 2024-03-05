import numpy as np
import pandas as pd
import openTSNE
from PIL import ImageFile

def opentsne_fit(features, n=2):
    tsne = openTSNE.TSNE(
        n_components=n,
        perplexity=30,
        initialization="pca",
        metric="cosine",
        random_state=0,
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tsne_results = tsne.fit(features)
    return tsne_results

def opentsne_transform(features, base_tsne):
    tsne_results = base_tsne.transform(features)
    return tsne_results


def compute_projections(features, path_images, predictions, labels, base_tsne=None, compute_base=False):
    if compute_base:
        base_tsne = opentsne_fit(features)
        projection = base_tsne.copy()
    else:
        projection = opentsne_transform(features, base_tsne)

    path_images = np.reshape(np.array(path_images), (-1, 1))
    predictions_arr = np.reshape(np.array(predictions), (-1, 1))
    labels_arr = np.reshape(np.array(labels), (-1, 1))
    tsne_arr = np.hstack((path_images, projection, predictions_arr, labels_arr))
    df = pd.DataFrame(tsne_arr, columns =['image_path', 'x', 'y', 'pred', 'label'])

    return base_tsne, df