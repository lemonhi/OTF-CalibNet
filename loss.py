"""Public loss functions for single-frame calibration."""
from calibration_objective import (
    DepthGradientLoss as GradientDifferenceLoss,
    SquaredChamferLoss as ChamferDistanceLoss,
    DirectParameterLoss as RTLoss,
    CalibrationObjective,
)
import torch
from utils import so3
from scipy.spatial.transform import Rotation

def geodesic_distance(transform):
    """Diagnostic transform errors; not the parameter training objective."""
    return (torch.linalg.vector_norm(so3.log(transform[:,:3,:3]),dim=1).mean(),
            torch.linalg.vector_norm(transform[:,:3,3],dim=1).mean())

def gt2euler(transform):
    """XYZ Euler angles and inverse-transform translation."""
    r=transform[:3,:3]
    return Rotation.from_matrix(r).as_euler('xyz'), -r.T @ transform[:3,3]
