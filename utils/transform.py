from . import se3,so3
import torch
import numpy as np
from math import pi as PI
from collections.abc import Iterable

class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, max_deg, max_tran, mag_randomly=True, concat=False):
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        if self.randomly:
            deg = torch.rand(1).item()*self.max_deg
            tran = torch.rand(1).item()*self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran
        amp = deg * PI / 180.0  # deg to rad
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        t = torch.rand(1, 3) * tran

        # the output: twist vectors.
        R = so3.exp(w) # (N, 3) --> (N, 3, 3)
        G = torch.zeros(1, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        x = se3.log(G) # --> (N, 6)
        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [3,N] or [6,N]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        if self.concat:
            return torch.cat([se3.transform(g, p0[:3,:]),so3.transform(g[:,:3,:3], p0[3:,:])], dim=1)  # [1, 4, 4] x [6, N] -> [6, N]
        else:
            return se3.transform(g, p0)   # [1, 4, 4] x [3, N] -> [3, N]

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class UniformTransformSE3:
    def __init__(self, max_deg, max_tran, mag_randomly=True, concat=False):
        self.max_deg = max_deg
        self.max_tran = max_tran
        self.randomly = mag_randomly
        self.concat = concat
        self.gt = None
        self.igt = None

    def     generate_transform(self):
        # return: a twist-vector
        if self.randomly:
            deg = torch.rand(1).item()*self.max_deg
            tran = torch.rand(1).item()*self.max_tran
        else:
            deg = self.max_deg
            tran = self.max_tran
        amp = deg * PI / 180.0  # deg to rad
        w = (2*torch.rand(1, 3)-1) * amp
        t = (2*torch.rand(1, 3)-1) * tran

        # the output: twist vectors.
        R = so3.exp(w) # (N, 3) --> (N, 3, 3)
        G = torch.zeros(1, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        x = se3.log(G) # --> (N, 6)
        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [3,N] or [6,N]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        if self.concat:
            return torch.cat([se3.transform(g, p0[:3,:]),so3.transform(g[:,:3,:3], p0[3:,:])], dim=1)  # [1, 4, 4] x [6, N] -> [6, N]
        else:
            return se3.transform(g, p0)   # [1, 4, 4] x [3, N] -> [3, N]

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)

class DepthImgGenerator:
    def __init__(self,img_shape:Iterable,InTran:torch.Tensor,pcd_range:torch.Tensor,pooling_size=5):
        assert (pooling_size-1) % 2 == 0, 'pooling size must be odd to keep image size constant'
        self.pooling = torch.nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=(pooling_size-1)//2)
        # InTran (3,4) or (4,4)
        self.img_shape = img_shape
        self.InTran = torch.eye(3)[None,...]
        self.InTran[0,:InTran.size(0),:InTran.size(1)] = InTran  # [1,3,3]
        self.pcd_range = pcd_range  # (B,N)

    def transform(self,ExTran:torch.Tensor,pcd:torch.Tensor)->tuple:
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        H,W = self.img_shape
        B = ExTran.size(0)
        self.InTran = self.InTran.to(pcd.device)
        pcd = se3.transform(ExTran,pcd)  # [B,4,4] x [B,3,N] -> [B,3,N]
        proj_pcd = torch.bmm(self.InTran.repeat(B,1,1),pcd) # [B,3,3] x [B,3,N] -> [B,3,N]
        proj_x = (proj_pcd[:,0,:]/proj_pcd[:,2,:]).type(torch.long)
        proj_y = (proj_pcd[:,1,:]/proj_pcd[:,2,:]).type(torch.long)
        rev = ((proj_x>=0)*(proj_x<W)*(proj_y>=0)*(proj_y<H)*(proj_pcd[:,2,:]>0)).type(torch.bool)  # [B,N]
        batch_depth_img = torch.zeros(B,H,W,dtype=torch.float32).to(pcd.device)  # [B,H,W]
        # size of rev_i is not constant so that a batch-formed operdation cannot be applied
        for bi in range(B):
            rev_i = rev[bi,:]  # (N,)
            proj_xrev = proj_x[bi,rev_i]
            proj_yrev = proj_y[bi,rev_i]
            batch_depth_img[bi*torch.ones_like(proj_xrev),proj_yrev,proj_xrev] = self.pcd_range[bi,rev_i]
        return batch_depth_img.unsqueeze(1), pcd   # (B,1,H,W), (B,3,N)
    
    def __call__(self,ExTran:torch.Tensor,pcd:torch.Tensor):
        """transform pcd and project it to img

        Args:
            ExTran (torch.Tensor): B,4,4
            pcd (torch.Tensor): B,3,N

        Returns:
            tuple: depth_img (B,H,W), transformed_pcd (B,3,N)
        """
        assert len(ExTran.size()) == 3, 'ExTran size must be (B,4,4)'
        assert len(pcd.size()) == 3, 'pcd size must be (B,3,N)'
        return self.transform(ExTran,pcd)
    
def pcd_projection(img_shape:tuple,intran:np.ndarray,pcd:np.ndarray,range:np.ndarray):
    """project pcd into depth img

    Args:
        img_shape (tuple): (H,W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3xN)
        range (np.ndarray): (N,)

    Returns:
        u,v,r,rev: u,v,r (with rev) and rev
    """
    H,W = img_shape
    proj_pcd = intran @ pcd
    u,v,w = proj_pcd[0,:], proj_pcd[1,:], proj_pcd[2,:]
    u = np.asarray(u/w,dtype=np.int32)
    v = np.asarray(v/w,dtype=np.int32)
    rev = (0<=u)*(u<W)*(0<=v)*(v<H)*(w>0)
    u = u[rev]
    v = v[rev]
    r = range[rev]
    return u,v,r,rev

import numpy as np

def binary_projection(img_shape: tuple, intran: np.ndarray, pcd: np.ndarray):
    """project pcd on img (binary mode)

    Args:
        img_shape (tuple): (H, W)
        intran (np.ndarray): (3x3)
        pcd (np.ndarray): (3, N)

    Returns:
        u, v, rev: u, v (without rev filter) and rev
    """
    H, W = img_shape
    proj_pcd = intran @ pcd
    u, v, w = proj_pcd[0, :], proj_pcd[1, :], proj_pcd[2, :]

    # 将 w 为 0 的地方替换为 0.000001
    w = np.where(w == 0, 0.000001, w)

    # 使用np.where直接进行除法，并确保w是正数
    u = np.asarray(u / w, dtype=np.int32)
    v = np.asarray(v / w, dtype=np.int32)

    # 确保 u 和 v 在图像尺寸范围内
    rev = (0 <= u) * (u < W) * (0 <= v) * (v < H) * (w > 0)

    return u, v, rev

# 示例调用
# H, W = 图像的高度和宽度
# intran = 相机内参矩阵 (3x3)
# pcd = 点云数据 (3, N)
# u, v, rev = binary_project

def nptrans(pcd:np.ndarray,G:np.ndarray)->np.ndarray:
    R,t = G[:3,:3], G[:3,[3]]  # (3,3), (3,1)
    return R @ pcd + t