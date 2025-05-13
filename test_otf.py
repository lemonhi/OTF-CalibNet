import argparse
import os

import cv2
import numpy
import yaml
import torch
import pykitti
import numpy as np
import matplotlib
import time

matplotlib.use('agg')
from matplotlib import pyplot as plt
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset, KITTI_perturb
from mylogger import get_logger, print_highlight, print_warning
from OTF_CalibNet import *
import loss as loss_utils
import utils
import numpy as np
import scipy
from scipy import linalg
from torchvision.utils import save_image


def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config", type=str, default='config.yml')
    parser.add_argument("--dataset_path", type=str, default='../data')
    parser.add_argument("--skip_frame", type=int, default=3, help='skip frame of dataset')
    parser.add_argument("--pcd_sample", type=int, default=-1)  # -1 means total sample
    parser.add_argument("--max_deg", type=float, default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran", type=float, default=0.2)  # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly", type=bool, default=True)
    # dataloader
    parser.add_argument("--batch_size", type=int, default=1, choices=[1],
                        help='batch size of test dataloader must be 1')
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--pin_memory", type=bool, default=True,
                        help='set it to False if your CPU memory is insufficient')
    # schedule
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--pretrained", type=str, default='./checkpoint/0.2-0.4-0.4(10,0.2)_input(gra&depth)_best.pth')
    parser.add_argument("--log_dir", default='log/')
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/")
    parser.add_argument("--res_dir", type=str, default='res/')
    parser.add_argument("--name", type=str, default='')
    # setting
    parser.add_argument("--inner_iter", type=int, default=1, help='inner iter of calibnet')
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


def computegradients(depthmap):
    # 假设 depthmap 的形状是 (batchsize, channels, height, width)
    # 定义 Sobel 卷积核
    sobelx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=depthmap.dtype, device=depthmap.device).unsqueeze(0).unsqueeze(
        0)  # (1, 1, 3, 3)
    sobely = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=depthmap.dtype, device=depthmap.device).unsqueeze(0).unsqueeze(
        0)  # (1, 1, 3, 3)

    # 确保输入图像是单通道的
    depthmap_single_channel = depthmap.mean(dim=1, keepdim=True)  # 平均所有通道得到单通道图像

    # 计算梯度
    gradx = F.conv2d(depthmap_single_channel, sobelx, padding=1)
    grady = F.conv2d(depthmap_single_channel, sobely, padding=1)

    # 计算梯度幅值
    gradmagnitude = torch.sqrt(gradx ** 2 + grady ** 2)

    # 由于我们只计算了一个通道的梯度，所以不需要再次取平均
    return gradmagnitude


def test(args, chkpt: dict, test_loader,total_time=0):

    model = OTF_CalibNet(depth_scale=args.scale)
    device = torch.device(args.device)
    model.to(device)
    model.load_state_dict(chkpt['model'])
    model.eval()
    logger = get_logger('{name}-Test'.format(name=args.name), os.path.join(args.log_dir, args.name + '_test.log'),
                        mode='w')
    logger.debug(args)
    res_npy = np.zeros([len(test_loader), 6])
    print(len(test_loader))
    num = 0
    max_time=0
    min_time=100
    total_time=0
    data = pykitti.odometry(base_path='../data', sequence='00')
    calib = data.calib
    for i, batch in enumerate(test_loader):
        rgb_img = batch['img'].to(device)
        B = rgb_img.size(0)
        pcd_range = batch['pcd_range'].to(device)
        uncalibed_pcd = batch['uncalibed_pcd'].to(device)
        uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
        uncalibed_dep_img = uncalibed_depth_img.detach().cpu().numpy()[0]
        save_image(uncalibed_depth_img,'./depth_map/pic-{}.png'.format(i))
        InTran = batch['InTran'][0].to(device)
        igt = batch['igt'].to(device)
        img_shape = rgb_img.shape[-2:]
        depth_generator = utils.transform.DepthImgGenerator(img_shape, InTran, pcd_range, CONFIG['dataset']['pooling'])
        # model(rgb_img,uncalibed_depth_img)
        Tcl = torch.eye(4).repeat(B, 1, 1).to(device)
        uncalibed_pcd_temp = uncalibed_pcd
        for j in range(args.inner_iter):

            start_time = time.time()
            # 计算RGB梯度（假设rgb_img在整个过程中不变）
            rgb_gradient = computegradients(rgb_img)
            uncalibed_depth_img_gra = computegradients(uncalibed_depth_img)
            # 使用当前深度图像调用模型，预测位姿
            twist_rot, twist_tsl = model(rgb_gradient, uncalibed_depth_img_gra)
            iter_Tcl = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))
            uncalibed_depth_img, uncalibed_pcd_temp = depth_generator(iter_Tcl,uncalibed_pcd_temp)
            Tcl = Tcl.bmm(iter_Tcl)
            print(num,'Tcl',Tcl)
            end_time = time.time()
            a=end_time-start_time
            if a<min_time:
                min_time=a
            if a>max_time:
                max_time=a
            print("单次时间",a)
            total_time=total_time+a
            print("平均单次时间：",total_time/(i+1))
            print(num, 'Tcl', Tcl)
        num = num + 1
        uncalibed_pcd = uncalibed_pcd.cpu().detach().numpy().squeeze()
        shape = uncalibed_pcd.shape
        new_row = np.ones((1, shape[1]))
        uncalibed_pcd = np.vstack([uncalibed_pcd, new_row])
        print('i', i, 'igt', igt)
        dg = Tcl.bmm(igt)
        print(num, 'dg', dg)
        rot_dx, tsl_dx = loss_utils.gt2euler(dg.squeeze(0).cpu().detach().numpy())
        rot_dx = rot_dx.reshape(-1)
        tsl_dx = tsl_dx.reshape(-1)
        res_npy[i, :] = np.abs(np.concatenate([rot_dx, tsl_dx]))
        print(res_npy[i, :].mean().item())

        # extran1 = (calib.T_cam2_velo) * igt.detach().numpy()  # [4,4]
        extran1 = Tcl.cpu().detach().numpy().squeeze()
        extran = linalg.inv(igt.cpu().detach().numpy().squeeze())
        intran = calib.K_cam2  # [3,3]
        img = rgb_img.detach().cpu().numpy()[0]
        H, W = img.shape[1:3]
        print('H,w', H, W)
        #        print(extran1.shape, intran.shape)
        #        print(pcd.shape, img.shape)
        #        pcd[-1, :] = 1.0
        # 原RGB图

        plt.figure(figsize=(15.5125, 4.7), dpi=80, tight_layout=True)
        #plt.subplot(2, 2, 1)
        img = img.transpose((1, 2, 0))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig('./1/pic-{}.png'.format(i),bbox_inches='tight', pad_inches=0)


        # ground truth投影得到的
        pcd3 = uncalibed_pcd.squeeze()
        pcd3 = extran @ pcd3
        pcd3 = intran @ pcd3[:3, :]
        u2, v2, w2 = pcd3[0, :], pcd3[1, :], pcd3[2, :]
        u2 = u2 / w2
        v2 = v2 / w2
        rev2 = (0 <= u2) * (u2 < W) * (0 <= v2) * (v2 < H) * (w2 > 0)
        u2 = u2[rev2]
        v2 = v2[rev2]
        r2 = np.linalg.norm(pcd3[:, rev2], axis=0)
        #plt.subplot(2, 2, 2)
        plt.axis([0, W, H, 0])
        #img = img.transpose((1, 2, 0))
        plt.imshow(img)
        #plt.title('B',fontsize='xx-large',fontweight='heavy')
        plt.axis('off')
        plt.scatter([u2], [v2], c=[r2], cmap='rainbow_r', alpha=0.5, s=2)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.savefig('./2/pic-{}.png'.format(i),bbox_inches='tight', pad_inches=0)

        # 有干扰的点云不进行任何修正
        pcd1 = uncalibed_pcd
        # pcd1 = extran @ pcd1
        pcd1 = pcd1.squeeze()
        pcd1 = intran @ pcd1[:3, :]
        u, v, w = pcd1[0, :], pcd1[1, :], pcd1[2, :]
        u = u / w
        v = v / w
        rev = (0 <= u) * (u < W) * (0 <= v) * (v < H) * (w > 0)
        u = u[rev]
        v = v[rev]
        r = np.linalg.norm(pcd1[:, rev], axis=0)
        #plt.subplot(2, 2, 3)
        plt.figure(figsize=(15.5125, 4.7), dpi=80, tight_layout=True)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.axis([0, W, H, 0])
        plt.imshow(img)
        #plt.title('C',fontsize='xx-large',fontweight='heavy')
        plt.axis('off')
        plt.scatter([u], [v], c=[r], cmap='rainbow_r', alpha=0.5, s=2)
        plt.savefig('./3/pic-{}.png'.format(i),bbox_inches='tight', pad_inches=0)

        # 利用预测的外参Tcl对uncalibed_pcd进行修正后的点云
        pcd2 = uncalibed_pcd.squeeze()
        pcd2 = extran1 @ pcd2
        pcd2 = intran @ pcd2[:3, :]
        u1, v1, w1 = pcd2[0, :], pcd2[1, :], pcd2[2, :]
        u1 = u1 / w1
        v1 = v1 / w1
        rev1 = (0 <= u1) * (u1 < W) * (0 <= v1) * (v1 < H) * (w1 > 0)
        u1 = u1[rev1]
        v1 = v1[rev1]
        r1 = np.linalg.norm(pcd2[:, rev1], axis=0)
        #plt.subplot(2, 2, 4)
        plt.figure(figsize=(15.5125, 4.7), dpi=80, tight_layout=True)
        plt.axis([0, W, H, 0])
        plt.imshow(img)
        plt.axis('off')
        plt.scatter([u1], [v1], c=[r1], cmap='rainbow_r', alpha=0.5, s=2)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.savefig('./4/pic-{}.png'.format(i),bbox_inches='tight', pad_inches=0)
        plt.savefig('./img_total(inner_iter=1)/pic-{}.png'.format(i), bbox_inches='tight')
        logger.info('[{:05d}|{:05d}],mrot:{:.4f},mtran:{:.4f}'.format(i + 1, len(test_loader), np.degrees(res_npy[i, :3].mean().item()),res_npy[i,3:].mean().item()))
        logger.info(np.degrees(np.around(res_npy[i,:3],4)))
        logger.info(np.around(res_npy[i,3:],4))
        if i == 100:
            break
    np.save(os.path.join(os.path.join(args.res_dir, '{name}.npy'.format(name=args.name))), res_npy)
    logger.info('Angle error (deg): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*np.degrees(np.mean(res_npy[:100, :3], axis=0))))
    logger.info('Translation error (m): X:{:.4f},Y:{:.4f},Z:{:.4f}'.format(*np.mean(res_npy[:100, 3:], axis=0)))
    logger.info(np.mean(np.degrees(np.mean(res_npy[:100,0:3]))))
    logger.info(np.mean(np.mean(res_npy[:100,3:])))
    print(max_time)
    print(min_time)

if __name__ == "__main__":

    args = options()
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    os.makedirs(args.log_dir, exist_ok=True)
    with open(args.config, 'r') as f:
        CONFIG: dict = yaml.load(f, yaml.SafeLoader)
    if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
        chkpt = torch.load(args.pretrained)
        CONFIG.update(chkpt['config'])
        update_args = ['resize_ratio', 'name', 'scale']
        for up_arg in update_args:
            setattr(args, up_arg, chkpt['args'][up_arg])
    else:
        raise FileNotFoundError('pretrained checkpoint {:s} not found!'.format(os.path.abspath(args.pretrained)))
    print_highlight('args have been received, please wait for dataloader...')

    test_split = [str(index).rjust(2, '0') for index in CONFIG['dataset']['test']]
    test_dataset = BaseKITTIDataset(args.dataset_path, args.batch_size,['11'], CONFIG['dataset']['cam_id'],
                                    skip_frame=args.skip_frame, voxel_size=CONFIG['dataset']['voxel_size'],
                                    pcd_sample_num=args.pcd_sample, resize_ratio=args.resize_ratio,
                                    extend_ratio=CONFIG['dataset']['extend_ratio']
                                    )
    os.makedirs(args.res_dir, exist_ok=True)
    test_perturb_file = os.path.join(args.checkpoint_dir, "test_seq_10_0.2.csv")
    test_length = len(test_dataset)
    if not os.path.exists(test_perturb_file):
        print_highlight("test pertub file dosen't exist, create one.")
        transform = utils.transform.UniformTransformSE3(args.max_deg, args.max_tran, args.mag_randomly)
        perturb_arr = np.zeros([test_length, 6])
        for i in range(test_length):
            perturb_arr[i, :] = transform.generate_transform().cpu().numpy()
            perturb_arr[i,:]=np.around(perturb_arr[i,:],4)
            print(perturb_arr[i, :])
        np.savetxt(test_perturb_file, perturb_arr, delimiter=',')
    else:  # check length
        test_seq = np.loadtxt(test_perturb_file, delimiter=',')
        if test_length != test_seq.shape[0]:
            print_warning('Incompatiable test length {}!={}'.format(test_length, test_seq.shape[0]))
            transform = utils.transform.UniformTransformSE3(args.max_deg, args.max_tran, args.mag_randomly)
            perturb_arr = np.zeros([test_length, 6])
            for i in range(test_length):
                perturb_arr[i, :] = transform.generate_transform().cpu().numpy()
            np.savetxt(test_perturb_file, perturb_arr, delimiter=',')
            print_highlight('test perturb file rewritten.')

    test_dataset = KITTI_perturb(test_dataset, args.max_deg, args.max_tran, args.mag_randomly,
                                 pooling_size=CONFIG['dataset']['pooling'], file=test_perturb_file)
    test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=0, pin_memory=args.pin_memory)
    total_time=0
    test(args, chkpt, test_dataloader)
