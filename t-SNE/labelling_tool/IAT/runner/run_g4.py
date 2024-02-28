import shutil
from random import shuffle
from math import ceil
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists
from knn_labeling import run_knn
import pandas as pd

def check_input(text, max_num):
    try:    
        assert int(text) > 0
        assert int(text) <= max_num
        return True
    except:
        print('Batch size should be an integer between 1 and', max_num)
        

def check_case(text, dataframes_path = None):
    if text == 'ALL':
        for i in range(num_batches):
            prepare_data('dataframes/batch{:04d}'.format(i+1), join(batches_path, 'batch{:04d}'.format(i+1)))
            print('\nBatch', i+1, 'prepared.\n')
        return False
    elif text == 'SKIP':
        return False
    else:
        while not check_input(text, num_batches):
            text = input('\nChoose batch number for feature extraction: ')
        batch_id = int(text)

        prepare_data(dataframes_path + 'batch{:04d}'.format(batch_id), join(batches_path, 'batch{:04d}'.format(batch_id)))
        
        return True

def show_status(batches_path, dataframes_path):
    batch1_is_labeled = False

    batches_list = listdir(batches_path)
    batches_list.sort()
    print()
    for batch in batches_list:
        batch_file = dataframes_path + batch + '.csv'
        if isfile(batch_file):
            df = pd.read_csv(batch_file)
            not_labeled = 0
            value_counts = df['colors'].value_counts()
            if 0 in value_counts:
                not_labeled = value_counts[0]
            num_rows = len(df)
            num_labeled = num_rows-not_labeled
            
            if not_labeled == 0:
                print(batch, '-> features are already extracted. All ' + str(num_rows) + ' images are labeled.')
                if batch == 'batch0001':
                    batch1_is_labeled = True
            else:
                print(batch, '-> features are already extracted. ' + str(num_labeled) + ' out of ' + str(num_rows) + ' images are labeled.')
               
        else:
            print(batch, '-> needs features extraction.')
    return batch1_is_labeled

projects_path = 'main/assets/'

project_name = 'lroot_g4'

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
path_to_csv = dataframes_path + 'batch{:04d}'.format(batch_id) + '_g4.csv'

system('python main/app.py ' + path_to_images + ' ' + path_to_images + ' ' + path_to_csv)