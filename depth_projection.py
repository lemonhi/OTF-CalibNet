"""Camera-z depth with bilinear splatting and 5x5 semi-dense pooling.

Bilinear splatting provides coordinate gradients for depth supervision.
Overlapping contributions are averaged. This rasterization convention is
explicit because subpixel and occlusion handling are not specified by formulas.
"""
import torch
import torch.nn.functional as F

def project_depth(points, intrinsics, image_shape, pooling_size=5):
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None].expand(points.shape[0],-1,-1)
    h, w = image_shape
    camera = intrinsics @ points
    z = points[:,2]
    divisor = camera[:,2].clamp_min(1e-6)
    u, v = camera[:,0]/divisor, camera[:,1]/divisor
    u0, v0 = u.floor(), v.floor()
    depths, weights = [], []
    for b in range(points.shape[0]):
        numerator = points.new_zeros(h*w)
        denominator = points.new_zeros(h*w)
        for dx, dy in ((0,0),(1,0),(0,1),(1,1)):
            x, y = u0[b]+dx, v0[b]+dy
            weight = (1-(u[b]-x).abs())*(1-(v[b]-y).abs())
            valid = (z[b]>1e-6)&(x>=0)&(x<w)&(y>=0)&(y<h)
            index = (y[valid].long()*w+x[valid].long())
            numerator = numerator.scatter_add(0,index,weight[valid]*z[b,valid])
            denominator = denominator.scatter_add(0,index,weight[valid])
        depths.append((numerator/denominator.clamp_min(1e-6)).reshape(h,w))
    depth = torch.stack(depths)[:,None]
    if pooling_size != 1:
        depth = F.max_pool2d(depth,pooling_size,stride=1,padding=pooling_size//2)
    return depth
