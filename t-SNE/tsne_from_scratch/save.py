import os
import numpy as np
import pandas as pd
from defaults import defaults
import matplotlib.pyplot as plt
import PIL
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


def save_csv(output_path, batch_id, result_df):
    dataframes_folder = os.path.join(output_path, defaults['dataframes'])
    if not os.path.isdir(dataframes_folder):
        os.mkdir(dataframes_folder, mode=0o755)
    result_df.to_csv(os.path.join(dataframes_folder, batch_id + '_' + '.csv'), index=None)
    
    
def get_image(path, paint=False, color=(1, 1, 1), zoom=0.2, dim=255):
    img = PIL.Image.open(path).convert('RGBA')
    img = np.array(img)
    if paint:
        img[:,:,0] = np.uint8(img[:,:,0] * color[0])
        img[:,:,1] = np.uint8(img[:,:,1] * color[1])
        img[:,:,2] = np.uint8(img[:,:,2] * color[2])
        img[:,:,3] = dim
    img = PIL.Image.fromarray(img)

    return OffsetImage(img, zoom=zoom)
    
    
def map_of_images(df, xrange, yrange, output_path, zoom, fig_size=40):
    df_x = pd.to_numeric(df['x'])
    df_y = pd.to_numeric(df['y'])

    df_filtered = df[(df_x >= xrange[0]) & (df_x <= xrange[1]) & (df_y >= yrange[0]) & (df_y <= yrange[1])]

    x = df_filtered['x']
    y = df_filtered['y']
    images_paths = df_filtered['image_path']

    f = plt.figure(figsize=(fig_size, fig_size), frameon=False)
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.axis('off')
    f.add_axes(ax)
    ax.scatter(x, y, s=0)

    for xs, ys, path in zip(x, y, images_paths):
        ab = AnnotationBbox(get_image(path, zoom=zoom), (xs, ys), frameon=False, box_alignment=(0, 1))
        ax.add_artist(ab)

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    f.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(f)

        
def save_backgrounds(output_path, batch_id, results_df, range=100):
    backgrounds_dir = os.path.join(output_path, defaults['backgrounds'])
    fig_size = 40
    factor = defaults['map_factor'] # defaults: 2 tsne, 20 umap
    xrange = [-range, range]
    yrange = [-range, range]
    zoom = fig_size / (factor * (xrange[1] - xrange[0]))

    backgrounds_path = os.path.join(backgrounds_dir, batch_id + '_' + '.png')
    map_of_images(results_df, xrange, yrange, backgrounds_path, zoom, fig_size)


def save_scatter_plots(output_path, batch_id, df):
    # Create a color map based on unique labels
    unique_labels = df['label'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))

    # Plotting
    plt.scatter(df['x'], df['y'], c=df['label'].map(color_map), label=df['label'].unique())

    plt.legend(title='Label')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot by Label')
    plt.savefig('scatter_plot_by_label.png')