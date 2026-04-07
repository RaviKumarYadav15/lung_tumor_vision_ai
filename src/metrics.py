# src/metrics.py
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        smooth: A tiny value added to the denominator to prevent division by zero
                in healthy patients where both the prediction and target are 0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        
        dice_score = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1.0 - dice_score

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