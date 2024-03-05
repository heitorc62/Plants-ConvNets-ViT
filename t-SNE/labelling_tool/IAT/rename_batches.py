from os import listdir, rename
from os.path import join,  isfile, exists

project_names = ['verao_impar_12_20', 'verao_impar_01_21', 'verao_impar_02_21', 'verao_impar_03_21', 'verao_impar_04_21', 'verao_impar_05_21', 'verao_impar_06_21']
path = 'main/assets'

for project in project_names:
    project_path = join(path, project)
    if exists(project_path):
        dataframes_path = join(project_path, 'dataframes')
        dataframes = [f for f in listdir(dataframes_path) if f[:5] == 'batch']
        for dataframe in dataframes:
            new_name = dataframe[:11] + dataframe[17:]
            print(dataframe, '->', new_name)
            rename(join(dataframes_path, dataframe), join(dataframes_path, new_name))
        backgrounds_path = join(project_path, 'backgrounds')
        backgrounds = [f for f in listdir(backgrounds_path) if f[:5] == 'batch']
        for background in backgrounds:
            new_name = background[:11] + background[17:]
            print(background, '->', new_name)
            rename(join(backgrounds_path, background), join(backgrounds_path, new_name))
        rename(join(path, project), join(path, project[6:]))


