import torch
from torch import nn
import torch.nn.functional as F
from Modules import *

from torchvision import models



class Aggregation(nn.Module):
    def __init__(self,inplanes=1024+512,planes=96,final_feat=(5,2)):
        super(Aggregation,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes,out_channels=planes*8,kernel_size=3,stride=2,padding=1)
        self.bn1 = nn.BatchNorm2d(planes*8)
        self.conv2 = nn.Conv2d(in_channels=planes*8,out_channels=planes*8,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*8)
        self.conv3 = nn.Conv2d(in_channels=planes*8,out_channels=planes*4,kernel_size=(2,1),stride=2)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.conv4 = nn.Conv2d(in_channels=planes*4,out_channels=planes*2,kernel_size=(2,1),stride=2)
        self.bn4 = nn.BatchNorm2d(planes*2)

        self.tr_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.tr_bn = nn.BatchNorm2d(planes)
        self.rot_conv = nn.Conv2d(in_channels=planes*2,out_channels=planes,kernel_size=1,stride=1)
        self.rot_bn = nn.BatchNorm2d(planes)

        self.tr_drop = nn.Dropout2d(p=0.7)
        self.rot_drop = nn.Dropout2d(p=0.7)
        self.tr_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.rot_pool = nn.AdaptiveAvgPool2d(output_size=final_feat)
        self.fc1 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
        self.fc2 = nn.Linear(planes*final_feat[0]*final_feat[1],3)  # 96*10
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in',nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()




    def forward(self,x:torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x_tr = self.tr_conv(x)
        x_tr = self.tr_bn(x_tr)
        x_tr = self.tr_drop(x_tr)
        x_tr = self.tr_pool(x_tr)  # (19,6)
        x_tr = self.fc1(x_tr.view(x_tr.shape[0],-1))
        x_rot = self.rot_conv(x)
        x_rot = self.rot_bn(x_rot)
        x_rot = self.rot_drop(x_rot)
        x_rot = self.rot_pool(x_rot)  # (19.6)
        x_rot = self.fc2(x_rot.view(x_rot.shape[0],-1))
        return x_rot, x_tr




class OTF_CalibNet(nn.Module):
    def __init__(self, backbone_pretrained=False, depth_scale=1.0, backbone_checkpoint=None):
        super().__init__()
        self.rgb_resnet = resnet22(inplanes=1, planes=64)
        self.depth_resnet = resnet22(inplanes=1, planes=32)
        self.aggregation = Aggregation(inplanes=1024+512, planes=96)
        # Inputs are already normalized Canny/Laplacian maps. No extra pooling
        # or metric-depth scaling is performed inside the feature network.
        if backbone_pretrained:
            if backbone_checkpoint is None:
                raise ValueError("Provide the RGB backbone checkpoint explicitly")
            self.load_rgb_backbone(backbone_checkpoint)

    def load_rgb_backbone(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        own = self.rgb_resnet.state_dict()
        mapped = {}
        for key, value in state.items():
            for prefix in ("module.", "backbone.", "rgb_resnet."):
                if key.startswith(prefix): key = key[len(prefix):]
            if key == "stem.0.weight" and value.shape[1] == 3:
                value = value.mean(1, keepdim=True)
            if key in own and value.shape == own[key].shape:
                mapped[key] = value
        missing = [key for key in own if not key.startswith("layer5.")
                   and not key.endswith("num_batches_tracked") and key not in mapped]
        if missing:
            raise ValueError("RGB checkpoint does not cover the base backbone: " + str(missing[:5]))
        self.rgb_resnet.load_state_dict(mapped, strict=False)
        # No subsequent initialization: the loaded RGB weights are preserved.

    def forward(self, rgb, depth):
        rgb_features = self.rgb_resnet(rgb)[-1]
        depth_features = self.depth_resnet(depth)[-1]
        return self.aggregation(torch.cat((rgb_features, depth_features), dim=1))
