import unittest
from types import SimpleNamespace
import torch
import numpy as np
from calibration_objective import DirectParameterLoss, DepthGradientLoss, SquaredChamferLoss, correction_matrix
from depth_projection import project_depth
from otf_preprocessing import prepare_inputs
from utils import se3

torch.set_num_threads(2)

class CalibrationTests(unittest.TestCase):
    def test_direct_inverse_target(self):
        # Cartesian translation differs from the translational SE(3) twist.
        r=torch.tensor([[0.,0.,.2]])
        t=torch.tensor([[.2,-.1,.05]])
        correction=correction_matrix(r,t)
        disturbance=torch.linalg.inv(correction)
        loss=DirectParameterLoss()(r,t,disturbance)
        self.assertLess(loss.item(),1e-6)
        self.assertGreater(DirectParameterLoss()(-r,-t,disturbance).item(),.1)

    def test_gradient_preserves_sign(self):
        a=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)
        # Magnitude-only comparison incorrectly gives zero for a versus -a.
        self.assertGreater(DepthGradientLoss()(a,-a).item(),0)
        self.assertEqual(DepthGradientLoss()(a,a).item(),0)

    def test_chamfer_squared_sum(self):
        a=torch.tensor([[[0.,2.],[0.,0.],[0.,0.]]],requires_grad=True)
        b=torch.tensor([[[0.,5.],[0.,0.],[0.,0.]]])
        value=SquaredChamferLoss(1)(a,b)
        self.assertAlmostEqual(value.item(),13.)
        value.backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_depth_loss_reaches_pose(self):
        cloud=torch.tensor([[[.2,.6,-.2],[.3,-.1,.4],[2.,2.2,2.4]]])
        k=torch.tensor([[8.,0.,8.],[0.,8.,8.],[0.,0.,1.]])
        r=torch.tensor([[.01,.02,-.01]],requires_grad=True)
        t=torch.tensor([[.02,-.01,.03]],requires_grad=True)
        transform=correction_matrix(r,t)
        moved=transform[:,:3,:3]@cloud+transform[:,:3,3,None]
        value=DepthGradientLoss()(project_depth(moved,k,(16,16)),project_depth(cloud,k,(16,16)))
        value.backward()
        for gradient in (r.grad,t.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(),0)

    def test_inputs_batch_one_and_two(self):
        args=SimpleNamespace(rgb_encoding='zero_one',canny_low=100.,canny_high=200.)
        for batch in (1,2):
            rgb=torch.zeros(batch,3,16,16);rgb[:,:,:,8:]=1
            depth=torch.zeros(batch,1,16,16);depth[:,:,:,8:]=2
            edges,response=prepare_inputs(rgb,depth,args)
            for x in (edges,response):
                self.assertEqual(tuple(x.shape),(batch,1,16,16))
                self.assertTrue(torch.isfinite(x).all())
                self.assertGreater(x.sum().item(),0)
                self.assertGreaterEqual(x.min().item(),0)
                self.assertLessEqual(x.max().item(),1)

    def test_empty_depth(self):
        k=torch.eye(3)
        points=torch.tensor([[[0.],[0.],[-1.]]])
        self.assertEqual(project_depth(points,k,(8,8)).sum().item(),0)

if __name__=='__main__':unittest.main()
