import shutil
from random import shuffle
from math import ceil
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
from knn_labeling import run_knn

def check_input(text, max_num):
    try:    
        assert int(text) > 0
        assert int(text) <= max_num
        return True
    except:
        print('Batch size should be an integer between 1 and', max_num)

projects_path = 'main/assets/'

project_name = 'lroot'

dataframes_path = 'main/assets/' + project_name + '/dataframes/'
samples_path = 'main/assets/' + project_name + '/samples/' + project_name
images_path = 'assets/' + project_name + '/images/'
batches_path = join('main', images_path)

num_batches = len(listdir(batches_path))

text = input('\nChoose batch for labeling: ')
while not check_input(text, num_batches):
    text = input('\nChoose batch for labeling: ')
batch_id = int(text)

path_to_images = join(images_path, 'batch{:04d}'.format(batch_id), 'samples/')
path_to_csv = dataframes_path + 'batch{:04d}'.format(batch_id) + '.csv'

system('python main/app.py ' + path_to_images + ' ' + path_to_images + ' ' + path_to_csv)