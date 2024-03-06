import os
import torch
import numpy as np
import pandas as pd

def make_path(path):
    dir = os.path.dirname(path)
    if dir: 
        if not os.path.exists(dir):
            os.makedirs(dir)
            
def save_model(model, output_dir, working_mode):
    # Save the model
    model_path = os.path.join(output_dir, f"{working_mode}/model.pth")
    make_path(model_path)
    torch.save(model.state_dict(), model_path)
    
def save_val_acc_history(val_acc_hist, output_dir, working_mode):
    # Save the validation accuracy history
    hist_path = os.path.join(output_dir, f"{working_mode}/val_acc_history.csv")
    hist_np = np.array([h.item() for h in val_acc_hist])
    make_path(hist_path)
    np.savetxt(hist_path, hist_np, delimiter=",")
    
def save_val_loss_history(val_loss_hist, output_dir, working_mode):
    #Save the validation loss history
    loss_hist_np = np.array([h for h in val_loss_hist])
    loss_hist_path = os.path.join(output_dir, f"{working_mode}/val_loss_history.csv")
    make_path(loss_hist_path)
    np.savetxt(loss_hist_path, loss_hist_np, delimiter=",")
    
def save_train_acc_history(train_acc_hist, output_dir, working_mode):
    # Save the train accuracy history
    hist_np = np.array([h.item() for h in train_acc_hist])
    hist_path = os.path.join(output_dir, f"{working_mode}/train_acc_history.csv")
    make_path(hist_path)
    np.savetxt(hist_path, hist_np, delimiter=",")
    
def save_train_loss_history(train_loss_hist, output_dir, working_mode):
    #Save the train loss history
    loss_hist_np = np.array([h for h in train_loss_hist])
    loss_hist_path = os.path.join(output_dir, f"{working_mode}/train_loss_history.csv")
    make_path(loss_hist_path)
    np.savetxt(loss_hist_path, loss_hist_np, delimiter=",")
    
def save_confusion_matrix(best_true, best_preds, output_dir, working_mode):
    # Convert lists to DataFrames
    confusion_df = pd.DataFrame({'True': best_true, 'Predicted': best_preds})

    # Save to csv
    confusion_path = os.path.join(output_dir, f"{working_mode}/confusion.csv")
    make_path(confusion_path)
    confusion_df.to_csv(confusion_path, index=False)