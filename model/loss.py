import torch
from torch import nn
from torch.nn import functional as F


class OBWeightedBCE(nn.Module):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, input, target):
        from segmentation_models_pytorch.losses._functional import soft_tversky_score

        #Structure of input: (model output)
        # [i, j, m, n]
        # [i, j, k, m, n]
        #  i : batch index
        #  j :
        #      0 - building is hot (model output)
        #  m : width
        #  n : height

        #Structure of the target: (ground truth)
        # [i, j, k, m, n]
        #  i : batch index
        #  j :
        #      0 - y1 image
        #      1 - y2 image
        #      2 - y3 (mixed target)
        #      3 - lambda value
        #  k : 
        #      0 - background is hot
        #      1 - weight map
        #  m : width
        #  n : height

        weight_map = target[:,2,1,:,:].unsqueeze(1)
        target_masks = target[:,2,0,:,:].unsqueeze(1)

        #reluOutput = F.relu6(input)
        sigmoidOutput = torch.sigmoid(input)
        thresholdOutput = (sigmoidOutput > self.threshold).type(torch.int)

        #Which pixels were predicted a bldg but aren't
        false_pos = (thresholdOutput - target_masks).clamp_min(0)

        #Only want to add extra loss to pixels that were predicted as bldgs but weren't
        FP_weights = (weight_map - false_pos).clamp_min(0)

        BCEOutput = F.binary_cross_entropy_with_logits(input, target_masks, reduction='none')

        weightedBCEOutput = (BCEOutput + FP_weights).mean()

        return weightedBCEOutput

class OpenBldgLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        from segmentation_models_pytorch.losses._functional import soft_tversky_score

        #Structure of input: (model output)
        # [i, j, m, n]
        # [i, j, k, m, n]
        #  i : batch index
        #  j :
        #      0 - building is hot (model output)
        #  m : width
        #  n : height

        #Structure of the target: (ground truth)
        # [i, j, k, m, n]
        #  i : batch index
        #  j :
        #      0 - y1 image
        #      1 - y2 image
        #      2 - y3 (mixed target)
        #      3 - lambda value
        #  k : 
        #      0 - buildings are hot
        #      1 - weight map
        #  m : width
        #  n : height

        # y1 image:
        weight_map = target[:,0,1,:,:]
        target_masks = target[:,0,0,:,:]

        reluOutput = F.relu(input, dim=1)
        argmaxOutput = reluOutput.argmax(1)

        #Which pixels were predicted a bldg but aren't
        false_pos = (argmaxOutput - target_masks[:,0,:,:]).clamp_min(0)
        #Only want to add extra loss to pixels that were predicted as bldgs but weren't
        FP_weights = (weight_map - false_pos).clamp_min(0)

        CEOutput = F.cross_entropy(input, target_masks, reduction='none')
        weightedCEOutput_1 = (CEOutput + FP_weights).mean()

        # y2 image:
        weight_map = target[:,1,1,:,:]
        target_masks = target[:,1,0,:,:]
        reluOutput = F.relu(input, dim=1)
        argmaxOutput = reluOutput.argmax(1)
        #Which pixels were predicted a bldg but aren't
        false_pos = (argmaxOutput - target_masks[:,0,:,:]).clamp_min(0)
        #Only want to add extra loss to pixels that were predicted as bldgs but weren't
        FP_weights = (weight_map - false_pos).clamp_min(0)
        
        CEOutput = F.binary_cross_entropy_with_logits(input, target_masks, reduction='none')
        weightedCEOutput_2 = (CEOutput + FP_weights).mean()

        lam_i = target[:,2,0,0,0]

        weightedCEOutput_tot = ((lam_i * weightedCEOutput_1) + ((1 - lam_i) * weightedCEOutput_2)).mean()

        return weightedCEOutput_tot

