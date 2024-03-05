import shutil
from random import shuffle
from math import ceil
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists, getmtime
from knn_labeling import run_knn
from time import strftime, localtime

def map_ids(images_path):
    map_id_to_batch = {}    
    l = listdir(join('main', images_path))
    l.sort()
    
    for i in range(len(l)):
        map_id_to_batch[i+1] = l[i]
    
    return map_id_to_batch

def print_batches(dataframes_path, map_id_to_batch, project_name):
    for key in map_id_to_batch.keys():
        csv_name = map_id_to_batch[key] + '_' + project_name + '.csv'
        mod_time = getmtime(join(dataframes_path, csv_name))
        mod_date = strftime('%Y-%m-%d %H:%M:%S', localtime(mod_time))
        
        print('batch ' + str(key) + ': \t' + csv_name + ' \tLast modification: ' + mod_date)


def check_input(text, max_num):
    try:    
        assert int(text) > 0
        assert int(text) <= max_num
        return True
    except:
        print('Batch size should be an integer between 1 and', max_num)

projects_path = 'main/assets/'

project_name = 'impar_04_21'

dataframes_path = 'main/assets/' + project_name + '/dataframes/'
samples_path = 'main/assets/' + project_name + '/samples/' + project_name
images_path = 'assets/' + project_name + '/images/'
thumbnails_path = 'assets/' + project_name + '/thumbnails/'
batches_path = join('main', images_path)

num_batches = len(listdir(batches_path))

map_id_to_batch = map_ids(images_path)
print_batches(dataframes_path, map_id_to_batch, project_name)

text = input('\nChoose batch for labeling: ')
while not check_input(text, num_batches):
    text = input('\nChoose batch for labeling: ')
batch_id = int(text)

path_to_images = join(images_path, map_id_to_batch[batch_id], 'samples/')
path_to_thumbnails = join(thumbnails_path, map_id_to_batch[batch_id], 'samples/')
path_to_csv = dataframes_path + map_id_to_batch[batch_id] + '_' + project_name + '.csv'

print(path_to_csv)

system('python main/app.py ' + path_to_images + ' ' + path_to_thumbnails + ' ' + path_to_csv + ' 100')