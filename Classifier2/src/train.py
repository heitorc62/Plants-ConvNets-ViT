import time
import copy
import torch


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
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('{} ---> Training complete in {:.0f}m {:.0f}s'.format(working_mode, time_elapsed // 60, time_elapsed % 60))
    print('{} ---> Best val Acc: {:4f}'.format(working_mode, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return {
                'model_ft': model, 
                'best_preds': best_preds, 
                'best_true': best_true, 
                'val_acc_history': val_acc_history, 
                'val_loss_history': val_loss_history, 
                'train_acc_history': train_acc_history, 
                'train_loss_history': train_loss_history
            }