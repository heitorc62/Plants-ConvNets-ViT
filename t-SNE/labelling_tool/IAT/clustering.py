import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import timeit
import numpy as np
import pandas as pd
#from sklearn.manifold import TSNE
from openTSNE import TSNE

#from umap import UMAP
import random
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import os
import time

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#assert len(sys.argv) >= 3

def read_model(model_name = 'inception', model_path = 'models/inception/model/', base_model_path = 'models/inception/base_model/'):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Reading model...')
    if model_name == 'inception': #InceptionResNetV2, num_features=1536
        img_height = 299  
        img_width = 299
        model = keras.models.load_model(model_path)
        base_model = keras.models.load_model(base_model_path)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    #print(current_time, 'Model read\n')
    return (img_height, img_width, model, base_model)

def get_final_model(model, base_model, layer_name = 'conv_7b_ac'):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Creating the final model...')
    layer_features = len(model.layers)-4
    
    model1 = Model(inputs=model.input, outputs=model.layers[layer_features-1].output)
    model2 = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    output = model2(model1.output)
    
    #features_list = [layer.output for layer in final_model.layers if layer.name in layer_names]
    #print(len(features_list))
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    output = global_average_layer(output)
    
    final_model = Model(model1.input, output)
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Final model created\n')
    return final_model

def prepare_images(images_path, img_width, img_height):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    #print(current_time, 'Preparing images...')

    test_gen = ImageDataGenerator()
    images = test_gen.flow_from_directory(
        directory=images_path,
        shuffle = False,
        class_mode=None,
        target_size=(img_height, img_width),
        batch_size=30,
    )
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Images created\n')

    return images

def compute_features(layer_name, model, base_model, images):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Computing features of ', layer_name, '...')

    final_model = get_final_model(model, base_model, layer_name)

    x = final_model.predict(images)

    image_names = []
    for filepath in images.filenames:
        image_names.append(filepath)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    #print(current_time, 'Features computed\n')

    return x, image_names

def read_csv(filename, feature_names):
    '''
        Input:
            filename: path to the csv file
            feature_names: list of the names of the features 
        Output:
            names: list with the paths to the images
            features: values of the features
    '''
    df = pd.read_csv(filename)
        
    names = df['names'].values

    features = df[feature_names].values
    
    return names, features

def compute_tsne(features, n=2, base_tsne = []):            
    #tsne = TSNE(n_components=n)
    #return tsne.fit_transform(features)
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    #print(current_time, 't-SNE starting...')

    if base_tsne == []:
        tsne = TSNE(
            n_components=n,
            perplexity=30,
            initialization="pca",
            metric="cosine",
            n_jobs=8,
            random_state=3,
        )
    
        tsne_results = tsne.fit(features)

    else:
        tsne_results = base_tsne.transform(features)

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    #print(current_time, 't-SNE computed\n')

    return tsne_results

#def compute_umap(features, n=2):            
#    umap = UMAP(n_components=n)
#    return umap.fit_transform(features)

def save_csv(features, features2d, image_names, csv_path):
    header = ['names', 'x', 'y', 'custom_data', 'manual_label', 'correct_label',\
    'x2', 'y2', 'x3', 'y3', 'x4', 'colors', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

    basenames = [os.path.basename(name) for name in image_names]

    names_col = np.array(basenames).reshape(len(basenames), 1)

    ids = []
    for i in range(len(image_names)):
        ids.append(i)
    ids = np.array(ids).reshape(len(image_names), 1)

    manual_labels = []
    for i in range(len(image_names)):
        manual_labels.append('_')
    manual_labels = np.array(manual_labels).reshape((len(image_names), 1))

    correct_labels = []
    for i in range(len(image_names)):
        name = image_names[i]
        end = name.find('/')
        correct_labels.append(name[0:end])
    correct_labels = np.array(correct_labels).reshape((len(image_names), 1))
    zeros_6cols = np.zeros((len(image_names), 6))

    transposed_features = np.transpose(np.array(features))[0]

    data = np.hstack((names_col, features2d, ids, manual_labels, correct_labels, zeros_6cols, transposed_features))
    
    df = pd.DataFrame(data, columns = header)
    df.to_csv(csv_path + '.csv', index=False)
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    #print(current_time, csv_path + '.csv saved\n')

def generate_dataframe(csv_path, base_path):
    layer_names_1d = ['block17_1_ac', 'block17_6_ac', 'block17_11_ac', 'block17_16_ac', 'mixed_7a', 'block8_1_ac', 'conv_7b_ac']
    
    layer_name_2d = 'mixed_7a'
#    feature_names = ['Layer_A', 'Layer_B', 'Layer_C', 'Layer_D', 'Layer_E', 'Layer_F', 'Layer_G']
    img_height, img_width, model, base_model = read_model()
    base_images = prepare_images(base_path, img_width, img_height)    

    base_features, image_names = compute_features(layer_name_2d, model, base_model, base_images)
    
    base_tsne = compute_tsne(base_features, 2)
    
    tsnes = []
    for layer_name in layer_names_1d:
        features_1d, _ = compute_features(layer_name, model, base_model, base_images)
        tsne_1d = compute_tsne(features_1d, 1)
        tsnes.append(tsne_1d)
        #print(tsne_1d.shape)
    
    save_csv(tsnes, base_tsne, image_names, csv_path)

def generate_transformed_dataframe(csv_path, base_path, transformed_path):
    layer_names_1d = ['block17_1_ac', 'block17_6_ac', 'block17_11_ac', 'block17_16_ac', 'mixed_7a', 'block8_1_ac', 'conv_7b_ac']
    
    layer_name_2d = 'mixed_7a'
#    feature_names = ['Layer_A', 'Layer_B', 'Layer_C', 'Layer_D', 'Layer_E', 'Layer_F', 'Layer_G']
    img_height, img_width, model, base_model = read_model()
    base_images = prepare_images(base_path, img_width, img_height)    
    transformed_images = prepare_images(transformed_path, img_width, img_height)    

    base_features, _ = compute_features(layer_name_2d, model, base_model, base_images)
    transformed_features, image_names = compute_features(layer_name_2d, model, base_model, transformed_images)
    #np.savetxt(layer_name + '_base.csv', features, delimiter=',')
    
    base_tsne = compute_tsne(base_features, 2)
    transformed_tsne = compute_tsne(transformed_features, 2, base_tsne)
    
    tsnes = []
    for layer_name in layer_names_1d:
        features_1d, _ = compute_features(layer_name, model, base_model, transformed_images)
        #np.savetxt(layer_name + '_1d.csv', data, delimiter=',')
        tsne_1d = compute_tsne(features_1d, 1)
        tsnes.append(tsne_1d)
        #print(tsne_1d.shape)
    
    save_csv(tsnes, transformed_tsne, image_names, csv_path)

def prepare_data(csv_path, batch_path, transformed_path = ''):
    if transformed_path == '':
        generate_dataframe(csv_path, batch_path)
    else:
        generate_transformed_dataframe(csv_path, batch_path, transformed_path)

#if __name__ == '__main__':
#    csv_path = sys.argv[1]
#    base_path = sys.argv[2]
#    
#    if len(sys.argv) == 3:
#        generate_dataframe(csv_path, base_path)
#    else:
#        transformed_path = sys.argv[3]
#        generate_transformed_dataframe(csv_path, base_path, transformed_path)
