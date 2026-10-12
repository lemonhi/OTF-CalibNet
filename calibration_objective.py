"""Depth, point-cloud and correction-parameter supervision."""
import torch
from torch import nn
import torch.nn.functional as F
from utils import so3

IMPLEMENTATION_VERSION = "structural_calibration_v2"

def correction_matrix(rotation, translation):
    """Axis-angle rotation in radians; Cartesian translation in metres."""
    top = torch.cat((so3.exp(rotation), translation.unsqueeze(-1)), dim=-1)
    bottom = rotation.new_zeros((rotation.shape[0], 1, 4))
    bottom[:, 0, 3] = 1
    return torch.cat((top, bottom), dim=1)

class DirectParameterLoss(nn.Module):
    def forward(self, rotation, translation, perturbation):
        # Dataset perturbation maps the reference cloud to the disturbed cloud.
        correction = torch.linalg.inv(perturbation)
        target_r = so3.log(correction[:, :3, :3])
        target_t = correction[:, :3, 3]
        return ((rotation-target_r).abs().sum(1) +
                (translation-target_t).abs().sum(1)).mean()

class DepthGradientLoss(nn.Module):
    def forward(self, corrected, reference):
        # Signed Sobel derivatives implement the two spatial derivatives.
        k = corrected.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]
        total = corrected.new_zeros(())
        for kernel in (k, k.transpose(-1,-2)):
            delta = F.conv2d(corrected-reference, kernel, padding=1)
            total = total + delta.abs().flatten(1).sum(1).mean()
        return total

class SquaredChamferLoss(nn.Module):
    def __init__(self, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size

    def _direction(self, source, target):
        # Search without retaining O(N*M) autograd state; gather matched points
        # to differentiate exact squared distances to the selected neighbours.
        pieces = []
        for part in source.split(self.chunk_size, dim=1):
            with torch.no_grad():
                distance = torch.cdist(part, target).square()
                nearest = distance.argmin(-1)
            matched = target.gather(1, nearest[...,None].expand(-1,-1,3))
            pieces.append((part-matched).square().sum(-1).sum(-1))
        return torch.stack(pieces).sum(0)

    def forward(self, reference, corrected):
        a, b = reference.transpose(1,2), corrected.transpose(1,2)
        return (self._direction(a,b)+self._direction(b,a)).mean()

class CalibrationObjective(nn.Module):
    def __init__(self, weights=(0.3,0.3,0.4)):
        super().__init__()
        self.weights = weights
        self.gradient = DepthGradientLoss()
        self.cloud = SquaredChamferLoss()
        self.direct = DirectParameterLoss()

    def forward(self, rotation, translation, perturbation,
                corrected_depth, reference_depth, corrected_cloud, reference_cloud):
        terms = (self.gradient(corrected_depth, reference_depth),
                 self.cloud(reference_cloud, corrected_cloud),
                 self.direct(rotation, translation, perturbation))
        return sum(w*x for w,x in zip(self.weights,terms)), terms
