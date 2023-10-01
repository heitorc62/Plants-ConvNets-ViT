from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split
import pandas as pd
import os
import argparse
from modules.model import initialize_model, define_optimizer
from modules.train import train_model





def load_data(data_dir, input_size, batch_size, train_percent=0.8):
    
    transform = transforms.Compose([                                        # Define your transform
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)           # Load the dataset
    train_len = int(train_percent * len(dataset))                           # 80% for training
    val_len = len(dataset) - train_len                                      # 20% for validation
    train_set, val_set = random_split(dataset, [train_len, val_len])
    image_datasets = {'train': train_set, 'val': val_set}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'val']}

    return dataloaders_dict


def parse_arguments():
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

    return working_mode, feature_extract_bool, use_pretrained_bool, args.epochs


def save_model(working_mode, model_ft):
    # Save the model
    model_path = os.path.join(current_dir, f"{working_mode}/model.pth")
    torch.save(model_ft.state_dict(), model_path)

def save_statistics(current_dir, working_mode, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist):
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




if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.realpath(__file__))              # Get the directory of the current file
    data_dir = os.path.join(current_dir, "../../dataset/Borders_dataset")  # We assume the data is in ImageFolder format
    model_name = "vgg"
    num_classes = 39
    batch_size = 8
    train_percent = 0.8
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("The selected device is:", device)


    working_mode, feature_extract_bool, use_pretrained_bool, num_epochs = parse_arguments()
    model_ft, input_size = initialize_model(num_classes, feature_extract_bool, use_pretrained_bool)   # Initialize the model for this run
    dataloaders_dict = load_data(data_dir, input_size, batch_size, train_percent=train_percent)       # Load the data
    optimizer_ft = define_optimizer(model_ft, device, feature_extract_bool)                           # Define the optimizer
    criterion = nn.CrossEntropyLoss()                                                                 # Define the loss function
    model_ft, best_preds, best_true, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, working_mode, num_epochs=num_epochs)


    save_model(working_mode, model_ft)
    save_statistics(current_dir, working_mode, val_acc_hist, val_loss_hist, train_acc_hist, train_loss_hist)
    print(f"############################################## {working_mode} ###############################################################")