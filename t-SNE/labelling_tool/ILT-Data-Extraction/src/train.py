import torch
import torch.nn as nn
import os
from os.path import join, exists
import glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sys import argv
import pandas as pd
import numpy as np
import shutil
from timeit import default_timer as timer
import psutil
from focal_loss.focal_loss import FocalLoss
import random
import torchvision.transforms.functional as FV
import json
from pathlib import Path
from features import get_model, get_transforms, seed_everything

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda'
assert len(argv) > 4, 'Missing parameters. Parameters are: path_to_images output_path project_name path_to_JSON_file'

class CustomDataset(Dataset):
    def __init__(self, file_list, map_label, transform=None):
        self.file_list = file_list
        self.map_label = map_label
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)
        
        label = img_path.split("/")[-2]
        label_id = self.map_label[label]
        
        return img_transformed, label_id, img_path

class EarlyStopper:
    def __init__(self, patience=20, counter = 0, min_val_loss = float('inf'), max_val_acc = 0):
        self.patience = patience
        self.counter = counter
        self.min_val_loss = min_val_loss
        self.max_val_acc = max_val_acc

    def early_stop(self, model, val_metric, val, optimizer, scaler, save_path, save_name):
        checkpoint = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()}
        torch.save(checkpoint, join(save_path, 'model_' + save_name + '_last_epoch'))
        if val_metric == 'valid_loss' and val < self.min_val_loss:
            print(val_metric, val, ' > ', self.min_val_loss, '? (', self.counter, ')')
        
            self.min_val_loss = val
            self.counter = 0
            torch.save(checkpoint, join(save_path, 'model_' + save_name))

        elif val_metric in ['valid_acc', 'valid_macro'] and val > self.max_val_acc:
            print(val_metric, val, ' > ', self.max_val_acc, '? (', self.counter, ')')
        
            self.max_val_acc = val
            self.counter = 0
            torch.save(checkpoint, join(save_path, 'model_' + save_name))

        else:
            self.counter += 1
            print('No improvement - ', self.counter)
            if self.counter >= self.patience:
                return True
        return False

def compute_macro(matrix, div = -1): #div is used to manually input the number of "valid" classes
    avg_recall = 0
    avg_precision = 0
    macro = 0
    n = matrix.shape[0]
    for x in range(n):
        tp = matrix[x][x]
        recall = 0
        precision = 0
        f1 = 0

        sum1 = matrix[x].sum()
        sum2 = matrix[:,x].sum()
        if sum1 > 0:
            recall = tp/sum1
        if sum2 > 0:
            precision = tp/sum2
        if (recall + precision) > 0:
            f1 = 2*recall*precision/(recall + precision)
        avg_recall += recall
        avg_precision += precision
        macro += f1

    if div < 0:
        div = n
    
    return avg_recall/div, avg_precision/div, macro/div


def run(images_path, save_path, save_name, train_name = 'train', valid_name = 'valid', test_name = 'test', lr=0.00001, \
    valid_metric = 'valid_acc', num_epochs = 200, patience = 20, loss_name = 'ce', fgamma = 2, \
    seed = -1, div=-1, num_workers = 0, read_model = 'F'):

    if seed != -1:
        seed_everything(seed)
    use_amp = True
    
    print('File name:', save_name)

    train_dir = os.path.join(images_path, train_name)
    valid_dir = os.path.join(images_path, valid_name)
    test_dir = os.path.join(images_path, test_name)

    train_list = glob.glob(os.path.join(train_dir,'*','*'))
    valid_list = glob.glob(os.path.join(valid_dir,'*','*'))
    test_list = glob.glob(os.path.join(test_dir,'*','*'))

    map_label = {}
    labels = os.listdir(train_dir)
    labels.sort()
    label_id = 0
    img_size = 224
    batch_size = 30

    for c in labels:
        map_label[c] = label_id
        label_id += 1

    labels_ids = [i for i in range(len(labels))]
    
    train_transform = get_transforms()
    test_transform = get_transforms()

    train_data = CustomDataset(train_list, map_label, transform=train_transform)
    valid_data = CustomDataset(valid_list, map_label, transform=test_transform)
    test_data = CustomDataset(test_list, map_label, transform=test_transform)

    train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    num_classes = len(labels)

    model = get_model(load = True, num_classes = num_classes)
    criterion = nn.CrossEntropyLoss()
    if loss_name == 'focal':
        criterion = FocalLoss(gamma=fgamma)
        m = torch.nn.Softmax(dim=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    counter = 0
    if exists(join(save_path, 'model_{}'.format(save_name))):
        option = '3' 
        partial_model = False
        if exists(join(save_path, 'model_{}_last_epoch'.format(save_name))):
            partial_model = True

        if read_model != 'T' or not partial_model:
            print('Warning! Model with name', save_name, 'already exists!')
            
            options = ['0', '1']
            if partial_model:
                options.append('2')
            else:
                print("Note: model already fully trained. Impossible to resume training.")
            print()
            print('Choose an option below:')
            while option not in options:
                print('  1: Start training from scratch and overwrite the existing model')
                if '2' in options:
                    print('  2: Resume training of existing model')
                print('  0: Stop running this script.')
                option = input('Enter your option: ')
        if option == '2':
            read_model = 'T'
        if option == '0':
            exit(-1)
        
    with open(join(save_path, 'labels_' + save_name + '.json'), "w") as outfile: 
        json.dump(map_label, outfile)
    
    model.to(device)
    min_val_loss = float('inf')
    max_val_acc = 0
    if read_model == 'T' and exists(join(save_path, 'model_{}_last_epoch'.format(save_name))):
        dev = torch.cuda.current_device()
        checkpoint = torch.load(join(save_path, 'model_{}_last_epoch'.format(save_name)), map_location=lambda storage, loc: storage.cuda(dev))

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        
        df_stats = pd.read_csv(join(save_path, 'stats_{}.csv'.format(save_name)), index_col = None)
        epoch = df_stats.shape[0] + 1
        if valid_metric == 'valid_acc':
            col_name = 'Valid Loss'
            best_row = df_stats[col_name].idxmin()
            min_val_loss = df_stats[col_name].min()
        elif valid_metric == 'valid_acc':
            col_name = 'Valid Acc'
            best_row = df_stats[col_name].idxmax()
            max_val_acc = df_stats[col_name].max()
        else: #valid_metric == 'valid_macro'
            col_name = 'Valid Macro-F1'
            best_row = df_stats[col_name].idxmax()
            max_val_acc = df_stats[col_name].max()
        counter = epoch - best_row - 2
        prev_time = df_stats['Time'].iloc[epoch-2]
    else:
        epoch = 1
        prev_time = 0    
        df_stats = pd.DataFrame(columns=['Epoch', 'Time', 'VRAM', 'RAM', 'Train Acc', 'Train Loss', 'Train Recall', 'Train Precision', 'Train Macro-F1', 'Valid Acc', 'Valid Loss', 'Valid Recall', 'Valid Precision', 'Valid Macro-F1'])
        

    early_stopper = EarlyStopper(patience=patience, counter = counter, min_val_loss = min_val_loss, max_val_acc = max_val_acc)

    start = timer()

    while epoch <= num_epochs and early_stopper.counter < patience:
        train_acc = 0
        train_loss = 0
        train_matrix = np.zeros((num_classes, num_classes))
        for data, label, _ in tqdm(train_loader):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                if loss_name == 'focal':
                    output = m(output)
                loss = criterion(output, label)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            pred = output.argmax(dim=1)
            acc = (pred == label).float().sum()
            train_acc += acc.item()
            train_loss += loss.item()

            for i in range(label.shape[0]):
                l = label[i]
                p = pred[i]
                train_matrix[l][p] += 1

        train_acc/=len(train_data)
        train_loss /= len(train_loader)
        train_rec, train_prec, train_macro = compute_macro(train_matrix)

        valid_acc = 0
        valid_loss = 0
        valid_matrix = np.zeros((num_classes, num_classes))
        with torch.no_grad():
            for data, label, _ in tqdm(valid_loader):
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    data = data.to(device)
                    label = label.to(device)
                    output = model(data)
                    if loss_name == 'focal':
                        output = m(output)
                
                    loss = criterion(output, label)
                
                pred = output.argmax(dim=1)

                acc = (pred == label).float().sum()
                valid_acc += acc.item()
                valid_loss += loss.item()
                for i in range(label.shape[0]):
                    l = label[i]
                    p = pred[i]
                    valid_matrix[l][p] += 1
        valid_acc /=len(valid_data)
        valid_loss /= len(valid_loader)
        valid_rec, valid_prec, valid_macro = compute_macro(valid_matrix)

        gpu_mem_usage = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])/(1024*1024)
        cur = timer()
        time_diff = cur - start
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss/(1024*1024)
    
        df_row = pd.DataFrame([[epoch, time_diff + prev_time, gpu_mem_usage, ram_usage, train_acc, train_loss, train_rec, train_prec, train_macro, valid_acc, valid_loss, valid_rec, valid_prec, valid_macro]],\
            columns=['Epoch', 'Time', 'VRAM', 'RAM', 'Train Acc', 'Train Loss', 'Train Recall', 'Train Precision', 'Train Macro-F1', 'Valid Acc', 'Valid Loss', 'Valid Recall', 'Valid Precision', 'Valid Macro-F1'])
        df_stats = pd.concat([df_stats, df_row], ignore_index = True)
        df_stats.to_csv(join(save_path, 'stats_{}.csv'.format(save_name)), index = None)

        print('Epoch {}, time: {:.2f} mins, VRAM: {:.1f}MB, RAM: {:.1f}MB'.format(epoch, (time_diff+prev_time)/60, gpu_mem_usage, ram_usage))
        print('\ttrain acc: {:.5f}, loss: {:.5f}, rec: {:.5f}, prec: {:.5f}, macro: {:.5f}'.format(train_acc, train_loss, train_rec, train_prec, train_macro))
        print('\tvalid acc: {:.5f}, loss: {:.5f}, rec: {:.5f}, prec: {:.5f}, macro: {:.5f}'.format(valid_acc, valid_loss, valid_rec, valid_prec, valid_macro))
        epoch += 1

        if valid_metric == 'valid_loss':
            metric_value = valid_loss
        if valid_metric == 'valid_acc':
            metric_value = valid_acc
        if valid_metric == 'valid_macro':
            metric_value = valid_macro
        early_stopper.early_stop(model, valid_metric, metric_value, optimizer, scaler, save_path, save_name)
            
    test_acc = 0
    test_loss = 0
    test_matrix = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for data, label, _ in tqdm(test_loader):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                if loss_name == 'focal':
                    output = m(output)
                
                loss = criterion(output, label)
                
            pred = output.argmax(dim=1)
            acc = (pred == label).float().sum()
            test_acc += acc.item()
            test_loss += loss.item()
            for i in range(label.shape[0]):
                l = label[i]
                p = pred[i]
                test_matrix[l][p] += 1

    
    test_acc /=len(test_data)
    test_loss /= len(test_loader)
    test_rec, test_prec, test_macro = compute_macro(test_matrix, div)

    gpu_mem_usage = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0])/(1024*1024)
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss/(1024*1024)
    cur = timer()
    time_diff = cur - start
    
    print('test acc: {:.5f}, loss: {:.5f}, rec: {:.5f}, prec: {:.5f}, macro: {:.5f}'.format(test_acc, test_loss, test_rec, test_prec, test_macro))
    df_row = pd.DataFrame([['Test', time_diff, gpu_mem_usage, ram_usage, '', '', '', '', '', test_acc, test_loss, test_rec, test_prec, test_macro]], \
        columns=['Epoch', 'Time', 'VRAM', 'RAM', 'Train Acc', 'Train Loss', 'Train Recall', 'Train Precision', 'Train Macro-F1', 'Valid Acc', 'Valid Loss', 'Valid Recall', 'Valid Precision', 'Valid Macro-F1'])
    df_stats = pd.concat([df_stats, df_row], ignore_index = True)
    df_stats.to_csv(join(save_path, 'stats_{}.csv'.format(save_name)), index = None)
    np.savetxt(join(save_path, 'matrix_{}.csv'.format(save_name)), test_matrix,
              delimiter = ", ")
              
    os.remove(join(save_path, 'model_' + save_name + '_last_epoch'))

    print('=============== Finished ===============')
    print()

with open(argv[4]) as read_file:
    dc = json.load(read_file)
    
    images_path = argv[1]
    save_path = argv[2]
    save_name = argv[3]

    train_name = dc['trainFolder']
    valid_name = dc['validFolder']
    test_name = dc['testFolder']
    lr = dc['lr']
    valid_metric = dc['validMetric']
    num_epochs = dc['numEpochs']
    patience = dc['patience']
    loss_name = dc['lossName']
    focal_gamma = dc['focalGamma']
    seed = dc['seed']
    metric_num_classes = dc['numClasses']
    num_workers = dc['numWorkers']
    read_model = dc['readModel']
    
    if len(images_path) == 0:
        print('Error! Please include the path to the images in the JSON file. ')
    elif len(save_path) == 0:
        print('Error! Please include the output path in the JSON file. ')
    else:
        run(images_path, save_path, save_name, train_name, valid_name, test_name, lr, valid_metric, num_epochs, patience, loss_name, focal_gamma, seed, metric_num_classes, num_workers, read_model)

