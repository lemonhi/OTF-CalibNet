"""Shared Canny/Laplacian inputs. Outputs: float32 [B,1,H,W] in [0,1]."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PREPROCESSING_VERSION = 'canny_laplacian_v1'

def add_preprocessing_options(parser):
    parser.add_argument('--rgb_encoding', choices=['zero_one', 'zero_255', 'minus_one_one', 'imagenet'], default='zero_one', help='Must match dataset RGB normalization; channels must be RGB, not BGR.')
    parser.add_argument('--canny_low', type=float, default=100.0)
    parser.add_argument('--canny_high', type=float, default=200.0)


def rgb_edges(rgb, args):
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('RGB must be [B,3,H,W]')
    # OpenCV Canny is CPU/non-differentiable; transfers count in pipeline latency.
    x = rgb.detach().float().cpu()
    if args.rgb_encoding == 'zero_255':
        x = x / 255.0
    elif args.rgb_encoding == 'minus_one_one':
        x = (x + 1.0) / 2.0
    elif args.rgb_encoding == 'imagenet':
        x = x * x.new_tensor([.229, .224, .225])[None,:,None,None] + x.new_tensor([.485,.456,.406])[None,:,None,None]
    if not torch.isfinite(x).all() or x.min() < -1e-4 or x.max() > 1.0001:
        raise ValueError('RGB values incompatible with --rgb_encoding; check dataset.py')
    if not 0 <= args.canny_low < args.canny_high:
        raise ValueError('Require 0 <= canny_low < canny_high')
    arr = (x.clamp(0,1).permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
    maps = [cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), args.canny_low, args.canny_high, apertureSize=3, L2gradient=True) for im in arr]
    return torch.from_numpy(np.stack(maps)[:,None]).to(device=rgb.device, dtype=torch.float32)/255.0


def depth_laplacian(depth):
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError('Depth must be [B,1,H,W] or [B,H,W]')
    depth = depth.float()
    kernel = depth.new_tensor([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]])[None,None]
    response = F.conv2d(F.pad(depth,(1,1,1,1),mode='replicate'),kernel).abs()
    # Per-image magnitude normalization, including zeros representing missing depth.
    return response / response.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-6)


def prepare_inputs(rgb, depth, args):
    return rgb_edges(rgb,args), depth_laplacian(depth)
