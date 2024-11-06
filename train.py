# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:37:39 2019

@author: Keshik
"""
from torch import nn
from tqdm import tqdm
import torch
import gc
import os
from utils import get_ap_score, get_ap_score_alpha_beta
import numpy as np


import torch
import torch.nn as nn

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CustomCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits shape: (N, D * 2) - where the last dimension represents concatenated (a, b)
        # targets shape: (N, D) - binary indicators for each class (one-hot encoded)

        # Reshape logits to get a and b
        # a and b will have shape (N, D)
        a = logits[:, :logits.shape[1] // 2]  # First half corresponds to a
        b = logits[:, logits.shape[1] // 2:]   # Second half corresponds to b

        # Calculate p as a / (a + b) for each class
        p = a / (a + b)

        # Compute cross-entropy: -[target * log(p) + (1 - target) * log(1 - p)]
        cross_entropy_loss = -(
            targets * torch.log(p) + (1 - targets) * torch.log(1 - p)
        )

        # Sum over labels for each instance
        loss = cross_entropy_loss.sum(dim=1)  # Sum over the labels for each instance

        # Apply reduction
        if self.reduction == 'sum':
            return loss.sum()  # Sum all losses for the batch
        elif self.reduction == 'mean':
            return loss.mean()  # Average loss for the batch
        else:
            return loss  # 'none' reduction, return individual losses for each instance



class Type2NLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(Type2NLLLoss, self).__init__()
        self.reduction = reduction
        # self.epsilon = 1e-8  # Small constant to avoid log of zero

    def forward(self, logits, targets):
        # logits shape: (N, D * 2) - where the last dimension represents concatenated (a, b)
        # targets shape: (N, D) - binary indicators for each class

        # Reshape logits to get a and b
        # a and b will have shape (N, D)
        a = logits[:, :logits.shape[1] // 2] #+ self.epsilon  # First half corresponds to a
        b = logits[:, logits.shape[1] // 2:] #+ self.epsilon  # Second half corresponds to b

        # Calculate the log of the beta function using lgamma
        log_beta_a_b = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

        # Compute log-likelihood contributions only for present labels (where target is 1)
        log_likelihood_for_present = (
                torch.lgamma(targets + a) +
                torch.lgamma(1 - targets + b) -
                torch.lgamma(1 + a + b) -
                log_beta_a_b
        )

        # Negative log-likelihood
        nll = -log_likelihood_for_present

        # Sum over labels for each instance
        nll = nll.sum(dim=1)  # Sum over the labels for each instance

        if self.reduction == 'sum':
            return nll.sum()  # Sum all NLLs for the batch
        elif self.reduction == 'mean':
            return nll.mean()  # Average NLL for the batch
        else:
            return nll  # 'none'


def train_model(model, device, optimizer, scheduler, train_loader, valid_loader, save_dir, model_num, epochs, log_file):
    """
    Train a deep neural network model
    
    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  images dataloader
        valid_dataloader: validation images dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history
        
    Returns:
        Training history and Validation history (loss and average precision)
    """
    
    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0
    
    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        print("-------Epoch {}----------".format(epoch+1))
        log_file.write("Epoch {} >>".format(epoch+1))
        scheduler.step()
        
        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0
            
            # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            # m = torch.nn.Sigmoid()

            criterion = Type2NLLLoss()
            # criterion = CustomCrossEntropyLoss()


            if phase == 'train':
                model.train(True)  # Set model to training mode
                
                for data, target in tqdm(train_loader):
                    #print(data)
                    target = target.float()
                    data, target = data.to(device), target.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # output = model(data)

                    output_0 = model(data)
                    output = torch.nn.functional.softplus(output_0) + 1e-8

                    loss = criterion(output, target)
                    
                    # Get metrics here
                    running_loss += loss # sum up batch loss
                    running_ap += get_ap_score_alpha_beta(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu((output)).detach().numpy())

                    # Backpropagate the system the determine the gradients
                    loss.backward()
                    
                    # Update the paramteres of the model
                    optimizer.step()

                    # clear variables
                    del data, target, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    #print("loss = ", running_loss)
                    
                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_ = running_ap/num_samples
                
                print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
                    tr_loss_, tr_map_))
                
                log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
                    tr_loss_, tr_map_))
                
                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)

            else:
                model.train(False)  # Set model to evaluate mode
        
                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target in tqdm(valid_loader):
                        target = target.float()
                        data, target = data.to(device), target.to(device)

                        # output= model(data)

                        output_0 = model(data)
                        output = torch.nn.functional.softplus(output_0) + 1e-8
                        
                        loss = criterion(output, target)
                        
                        running_loss += loss # sum up batch loss
                        running_ap += get_ap_score_alpha_beta(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu((output)).detach().numpy())
                        
                        del data, target, output
                        gc.collect()
                        torch.cuda.empty_cache()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_map_ = running_ap/num_samples
                    
                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_map.append(val_map_)
                
                    print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
                    val_loss_, val_map_))
                    
                    log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
                    val_loss_, val_map_))
                    
                    # Save model using val_acc
                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        log_file.write("saving best weights...\n")
                        torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))
                    
    return ([tr_loss, tr_map], [val_loss, val_map])

    

def test(model, device, test_loader, returnAllScores=False):
    """
    Evaluate a deep neural network model
    
    Args:
        model: pytorch model object
        device: cuda or cpu
        test_dataloader: test images dataloader
        returnAllScores: If true addtionally return all confidence scores and ground truth 
        
    Returns:
        test loss and average precision. If returnAllScores = True, check Args
    """
    model.train(False)
    
    running_loss = 0
    running_ap = 0
    
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    # m = torch.nn.Sigmoid()

    criterion = Type2NLLLoss()
    # criterion = CustomCrossEntropyLoss()
    
    if returnAllScores == True:
        all_scores = np.empty((0, 20), float)
        ground_scores = np.empty((0, 20), float)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            #print(data.size(), target.size())
            target = target.float()
            data, target = data.to(device), target.to(device)
            bs, ncrops, c, h, w = data.size()

            output_0 = model(data.view(-1, c, h, w))
            output_1 = output_0.view(bs, ncrops, -1).mean(1)
            output = torch.nn.functional.softplus(output_1) + 1e-8

            loss = criterion(output, target)
            
            running_loss += loss # sum up batch loss
            running_ap += get_ap_score_alpha_beta(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu((output)).detach().numpy())
            
            if returnAllScores == True:

                all_scores = np.append(all_scores, torch.Tensor.cpu((output)).detach().numpy() , axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy() , axis=0)
            
            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item()/num_samples
    test_map = running_ap/num_samples
    
    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
                    avg_test_loss, test_map))
    
    
    if returnAllScores == False:
        return avg_test_loss, running_ap
    
    return avg_test_loss, running_ap, all_scores, ground_scores


