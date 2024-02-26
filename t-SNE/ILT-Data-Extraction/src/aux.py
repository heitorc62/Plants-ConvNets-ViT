import os

defaults = {
    'BATCH_MAX_SIZE': 8000,
    'output': 'output',
    'images': 'images',
    'dataframes': 'dataframes',
    'backgrounds': 'backgrounds',
    'thumbnails': 'thumbnails',
    'root': os.path.dirname(os.getcwd()),
    'output_folder': '',
    'inner_folder': 'samples',
    'thumbnails_size': 100,
    'pixel_size': 12.87,
    'ruler_ratio': 0.5,
    'base_tsne_id': 'batch_0001',
    'num_classes': 21,
    'map_factor': 2 # decrease to increase the size of the images in the backgrounds
}

def update_defaults():
  defaults['output_folder'] = os.path.join(defaults['root'], defaults['output'])

update_defaults()