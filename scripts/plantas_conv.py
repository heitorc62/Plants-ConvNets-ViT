from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
import time
import copy
import pandas as pd
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torchvision.models.vgg import VGG16_BN_Weights
import os
import argparse




def train_model(model, dataloaders, criterion, optimizer, device, working_mode, num_epochs=25):
    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # temporary variables to store predictions and true labels
        epoch_preds = []
        epoch_true = []

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # In train mode we calculate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # store predictions and true labels
                # we're only interested in validation performance
                if phase == 'val':  
                    epoch_preds.extend(preds.tolist())
                    epoch_true.extend(labels.data.tolist())



            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} ---> {} Loss: {:.4f} Acc: {:.4f}'.format(working_mode, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds, best_true = epoch_preds, epoch_true

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('{} ---> Training complete in {:.0f}m {:.0f}s'.format(working_mode, time_elapsed // 60, time_elapsed % 60))
    print('{} ---> Best val Acc: {:4f}'.format(working_mode, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_preds, best_true, val_acc_history, val_loss_history, train_acc_history, train_loss_history



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


    
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    # Defining the model as VGG:
    if use_pretrained:
        model_ft = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    else:
        model_ft = models.vgg16_bn()
    
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

    return model_ft, input_size




def load_data(data_dir, input_size, batch_size, train_percent=0.8):
    # Define your transform
    transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Determine the lengths of training and validation sets
    train_len = int(train_percent * len(dataset))  # 80% for training
    val_len = len(dataset) - train_len   # 20% for validation

    # Split the dataset
    train_set, val_set = random_split(dataset, [train_len, val_len])

    # Create a dictionary of the datasets
    image_datasets = {'train': train_set, 'val': val_set}

    # Create the dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    return dataloaders_dict


def define_optimizer(model_ft, device, feature_extract):
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized, i.e. the parameters with requires_grad
    # is True. Notice that, when finetuning, all the parameters have requires_grad = True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft



    





if __name__ == "__main__":

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model with different modes')

    # Add the arguments
    parser.add_argument('--epochs', type=int, help='The number of epochs')
    parser.add_argument('--feature_extract', type=str, default=False, help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params')
    parser.add_argument('--use_pretrained', type=str, default=False, help='Use pretrained model or not')

    # Parse the arguments
    args = parser.parse_args()
    
    feature_extract_bool = False
    if args.feature_extract == "True": feature_extract_bool = True
    use_pretrained_bool = False
    if args.use_pretrained == "True": use_pretrained_bool = True

    print("The selected epochs is:", args.epochs)
    print("The selected feature_extract is:", feature_extract_bool)
    print("The selected use_pretrained is:", use_pretrained_bool)
    
    working_mode = ""
    if feature_extract_bool and use_pretrained_bool:
        working_mode = "last_layer"
    elif not feature_extract_bool and use_pretrained_bool:
        working_mode = "fine_tuning"
    elif not feature_extract_bool and not use_pretrained_bool:
        working_mode = "from_scratch"

    print("The selected mode is:", working_mode)

    data_dir = os.path.join(current_dir, "../../dataset/Plant_leave_diseases_dataset_without_augmentation") # We assume the data is in ImageFolder format
    model_name = "vgg"
    num_classes = 39
    batch_size = 8 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("The selected device is:", device)

    # Initialize the model for this run
    model_ft, input_size = initialize_model(num_classes, feature_extract_bool, use_pretrained=use_pretrained_bool)
    # Print the model we just instantiated
    print(model_ft)

    # Load the data
    dataloaders_dict = load_data(data_dir, input_size, batch_size, train_percent=0.8)

    # Define the optimizer
    optimizer_ft = define_optimizer(model_ft, device, feature_extract_bool)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    # In order to produce matrics for the model, we will store confusion matrix necessary values.
    model_ft, best_preds, best_true, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, working_mode, num_epochs=args.epochs)



    # Save the model
    model_path = os.path.join(current_dir, f"{working_mode}/model.pth")
    torch.save(model_ft.state_dict(), model_path)

    # Save the validation accuracy history
    hist_np = np.array([h.item() for h in val_acc_hist])
    hist_path = os.path.join(current_dir, f"{working_mode}/val_acc_history.csv")
    np.savetxt(hist_path, hist_np, delimiter=",")

    #Save the validation loss history
    loss_hist_np = np.array([h for h in val_loss_hist])
    loss_hist_path = os.path.join(current_dir, f"{working_mode}/val_loss_history.csv")
    np.savetxt(loss_hist_path, loss_hist_np, delimiter=",")


    # Save the train accuracy history
    hist_np = np.array([h.item() for h in train_acc_hist])
    hist_path = os.path.join(current_dir, f"{working_mode}/train_acc_history.csv")
    np.savetxt(hist_path, hist_np, delimiter=",")

    #Save the train loss history
    loss_hist_np = np.array([h for h in train_loss_hist])
    loss_hist_path = os.path.join(current_dir, f"{working_mode}/train_loss_history.csv")
    np.savetxt(loss_hist_path, loss_hist_np, delimiter=",")

    # Convert lists to DataFrames
    confusion_df = pd.DataFrame({'True': best_true, 'Predicted': best_preds})

    # Save to csv
    confusion_path = os.path.join(current_dir, f"{working_mode}/confusion.csv")
    confusion_df.to_csv(confusion_path, index=False)
    print(f"############################################## {working_mode} ###############################################################")

