import multiprocessing as mp
import os
import sys
import timeit
from datetime import timedelta

import pandas as pd
import tqdm

from aux import defaults
from batch import create_batches, move_images
from data import (add_scale, generate_bkg, generate_thumbnails,
                  label_predictions, remove_scale)
from features import compute_features, get_model
from projections import compute_projections


def update(progress_bar):
    progress_bar.update(1)


def main():
    # read argv
    if len(sys.argv) > 2:
        output_path = os.path.abspath(sys.argv[1])
        project_name = os.path.basename(output_path)
        if len(sys.argv) == 4:
            weights_path = os.path.abspath(sys.argv[2])
            if not os.path.exists(weights_path):
                print('Error! Model file not found!')
                exit()
            labels_path = os.path.abspath(sys.argv[3])
            if not os.path.exists(labels_path):
                print('Error! Label file not found!')
                exit()
            print('Weights:', weights_path, "\n", 'Labels:', labels_path, "\n")
        else:
            weights_path, labels_path = '', ''

    model = get_model(load=True, num_classes=defaults['num_classes'])
    images_folder = os.path.join(output_path, defaults['images'])
    df_batches = pd.read_csv(os.path.join(output_path, 'batches.csv'), index_col=None)

    base_id = defaults['base_tsne_id']
    print('Computing base features...')
    start = timeit.default_timer()
    features, path_images, predictions = compute_features(images_folder, base_id, model, weights_path)
    end = timeit.default_timer()
    print('Total time:', timedelta(seconds=(end - start)), "\n")

    print('Computing base projections...')
    start = timeit.default_timer()
    base_tsne = compute_projections(output_path, project_name, base_id, features, path_images, df_batches, predictions, compute_base=True, save=False)
    end = timeit.default_timer()
    print('Total time:', timedelta(seconds=(end - start)), "\n")

    num_batches = len(os.listdir(images_folder))
    df_folder = os.path.join(output_path, defaults['dataframes'])

    # Step 2: Extract data
    print('Computing all features/projections...')
    for i in tqdm.trange(num_batches, ascii=True, ncols=79, unit='batch'):
        batch_id = 'batch_{:04d}'.format(i + 1)
        features, path_images, predictions = compute_features(images_folder, batch_id, model, weights_path)
        compute_projections(output_path, project_name, batch_id, features, path_images, df_batches, predictions, base_tsne=base_tsne)
    print()

    # Step 6: Label predictions
    if not labels_path == '':
        print('Labeling predictions...')
        start = timeit.default_timer()
        for i in tqdm.trange(num_batches, ascii=True, ncols=79, unit='batch'):
            batch_id = 'batch_{:04d}'.format(i + 1)
            label_predictions(df_folder, labels_path, project_name, batch_id)
        end = timeit.default_timer()
        print('Total time:', timedelta(seconds=(end - start)))


if __name__ == '__main__':
    main()