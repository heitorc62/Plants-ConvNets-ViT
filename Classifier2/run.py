from __future__ import print_function 
from __future__ import division
import torch
print("PyTorch Version: ",torch.__version__)
import torchvision
print("Torchvision Version: ",torchvision.__version__)
import argparse
from src.save import *
from src.data import load_data
from src.train import train_model
from src.model import *
    
    
    


def main(epochs, feature_extract, use_pretrained, working_mode,data_dir, output_dir):
    
    device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print("The selected device is:", device)
    # Initialize the model for this run
    model_ft, input_size = initialize_model(feature_extract, use_pretrained)
    # Print the model we just instantiated
    print(model_ft)
    # Load the data
    dataloaders_dict = load_data(data_dir, input_size)
    # Define the optimizer
    optimizer_ft = define_optimizer(model_ft, device, feature_extract_bool)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    # In order to produce matrics for the model, we will store confusion matrix necessary values.
    results = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, working_mode, num_epochs=epochs)

    save_model(results['model_ft'], output_dir, working_mode)
    save_val_acc_history(results['val_acc_hist'], output_dir, working_mode)
    save_val_loss_history(results['val_loss_hist'], output_dir, working_mode)
    save_train_acc_history(results['train_acc_hist'], output_dir, working_mode)
    save_train_loss_history(results['train_loss_hist'], output_dir, working_mode)
    save_confusion_matrix(results['best_true'], results['best_preds'], output_dir, working_mode)
    print(f"############################################## {working_mode} ###############################################################")


    





if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a model with different modes')

    # Add the arguments
    parser.add_argument('--epochs', type=int, help='The number of epochs')
    parser.add_argument('--feature_extract', type=str, default=False, help='Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params')
    parser.add_argument('--use_pretrained', type=str, default=False, help='Use pretrained model or not')
    parser.add_argument('--dataset_dir', type=str, default='/home/heitorc62/PlantsConv/dataset/Plant_leave_diseases_dataset_without_augmentation')
    parser.add_argument('--output_dir', type=str, default='/home/heitorc62/PlantsConv/Plants-ConvNets-ViT/Classifier2/results')

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
    
    main(args.epochs, feature_extract_bool, use_pretrained_bool, working_mode, args.dataset_dir, args.output_dir)
