from os import listdir, rename
from os.path import join,  isfile, exists

project_name = 'lroot_g4'
path = 'main/assets'

project_path = join(path, project_name)
if exists(project_path):
    dataframes_path = join(project_path, 'dataframes')
    dataframes = [f for f in listdir(dataframes_path) if f[:5] == 'batch']
    for dataframe in dataframes:
        new_name = dataframe[:-6] + project_name + '.csv'
        print(dataframe, '->', new_name)
        rename(join(dataframes_path, dataframe), join(dataframes_path, new_name))
    backgrounds_path = join(project_path, 'backgrounds')
    backgrounds = [f for f in listdir(backgrounds_path) if f[:5] == 'batch']
    for background in backgrounds:
        new_name = background[:-6] + project_name + '.png'
        print(background, '->', new_name)
        rename(join(backgrounds_path, background), join(backgrounds_path, new_name))
    

