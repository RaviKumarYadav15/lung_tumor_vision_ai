# src/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Combines Binary Cross Entropy (BCE) and Dice Loss.
        BCE is great for pixel-level classification, while Dice handles shape overlap.
        smooth: A tiny value added to prevent division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1. Calculate BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        
        # 2. Calculate Dice Loss
        inputs_sigmoid = torch.sigmoid(inputs)       
        
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                                            
        
        dice_score = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)  
        dice_loss = 1.0 - dice_score
        
        # 3. Combine them 
        return bce_loss + dice_loss

def calculate_iou(inputs, targets, smooth=1e-6):
    """
    Intersection over Union (IoU) / Jaccard Index.
    This is strictly for evaluating the model on our dashboard later, 
    we don't usually train the network with this metric.
    """
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float()
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    
    IoU = (intersection + smooth) / (union + smooth)
    return IoU