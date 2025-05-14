import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from utils import so3
from losses.chamfer_loss import chamfer_distance
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
    
class GradientDifferenceLoss(nn.Module):
    def __init__(self, scale=1.0, reduction='mean'):
        super(GradientDifferenceLoss, self).__init__()
        assert reduction in ['sum', 'mean', 'none'], 'Unknown or invalid reduction'
        self.scale = scale
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Gradient difference loss

        Args:
            input (torch.Tensor): (B, C, H, W)
            target (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: gradient difference loss between input and target
        """
        # Calculate gradients of input and target
        input_gradients = self.calculate_gradients(input)
        target_gradients = self.calculate_gradients(target)

        # Calculate the gradient difference
        gradient_difference = F.l1_loss(input_gradients, target_gradients, reduction=self.reduction)

        return gradient_difference * self.scale

    def calculate_gradients(self, tensor: torch.Tensor):
        # Create a kernel for gradient calculation in x and y directions
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)

        # Apply the kernel with padding to maintain the same size
        grad_x = F.conv2d(tensor, kernel_x, padding=1)
        grad_y = F.conv2d(tensor, kernel_y, padding=1)

        # Compute the gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        return grad_mag

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(input, target)


class Gradient_Loss(nn.Module):
    def __init__(self, scale=1.0):
        super(Gradient_Loss, self).__init__()
        self.scale = scale

    def calculate_gradient(self, image):
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate gradients using Sobel operator
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return gradient_magnitude

    def forward(self, image1, image2):
        # Convert PyTorch tensors to NumPy arrays
        image1_np = image1.cpu().numpy().transpose(1, 2, 0)
        image2_np = image2.cpu().numpy().transpose(1, 2, 0)

        # Calculate gradients
        grad1 = self.calculate_gradient(image1_np)
        grad2 = self.calculate_gradient(image2_np)

        # Convert gradients back to PyTorch tensors
        grad1_tensor = torch.from_numpy(grad1).unsqueeze(0).unsqueeze(0)
        grad2_tensor = torch.from_numpy(grad2).unsqueeze(0).unsqueeze(0)

        # Calculate gradient difference
        grad_diff = F.l1_loss(grad1_tensor, grad2_tensor)

        return grad_diff
class ChamferDistanceLoss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(ChamferDistanceLoss, self).__init__()
        assert reduction in ['sum','mean','none'], 'Unknown or invalid reduction'
        self.reduction = reduction
        self.scale = scale
    def forward(self, template, source):
        p0 = template/self.scale
        p1 = source/self.scale
        if self.reduction == 'none':
            return chamfer_distance(p0, p1)
        elif self.reduction == 'mean':
            return torch.mean(chamfer_distance(p0, p1),dim=0)
        elif self.reduction == 'sum':
            return torch.sum(chamfer_distance(p0, p1),dim=0)
    def __call__(self,template:torch.Tensor,source:torch.Tensor)->torch.Tensor:
        return self.forward(template,source)

class RTLoss(nn.Module):
    def __init__(self,scale=1.0,reduction='mean'):
        super(RTLoss,self).__init__()



def geodesic_distance(x:torch.Tensor,)->tuple:
        """geodesic distance for evaluation

        Args:
            x (torch.Tensor): (B,4,4)

        Returns:
            torch.Tensor(1),torch.Tensor(1): distance of component R and T
        """
        R = x[:,:3,:3]  # (B,3,3) rotation
        T = x[:,:3,3]  # (B,3) translation
        dR = so3.log(R) # (B,3)
        dR = F.mse_loss(dR,torch.zeros_like(dR).to(dR),reduction='none').mean(dim=1)  # (B,3) -> (B,1)
        dR = torch.sqrt(dR).mean(dim=0)  # (B,1) -> (1,)  Rotation RMSE (mean in batch)
        dT = F.mse_loss(T,torch.zeros_like(T).to(T),reduction='none').mean(dim=1) # (B,3) -> (B,1)
        dT = torch.sqrt(dT).mean(dim=0)  # (B,1) -> (1,) Translation RMSE (mean in batch)
        return dR, dT

def gt2euler(gt:np.ndarray):
    """gt transformer to euler anlges and translation

    Args:
        gt (np.ndarray): 4x4

    Returns:
        angle_gt, trans_gt: (3,1),(3,1)
    """
    R_gt = gt[:3, :3]
    euler_angle = Rotation.from_matrix(R_gt)
    anglez_gt, angley_gt, anglex_gt = euler_angle.as_euler('zyx')
    angle_gt = np.array([anglex_gt, angley_gt, anglez_gt])
    trans_gt_t = -R_gt @ gt[:3, 3]
    return angle_gt, trans_gt_t
