import argparse
from asyncio.log import logger
import os
import yaml
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset import BaseKITTIDataset,KITTI_perturb
from mylogger import get_logger, print_highlight, print_warning
from OTF_CalibNet import OTF_CalibNet
import loss as loss_utils
import utils
from tqdm import tqdm
import numpy as np
from utils.transform import UniformTransformSE3
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.nn.functional as F

def options():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--config",type=str,default='config.yml')
    parser.add_argument("--dataset_path",type=str,default='../data/')
    parser.add_argument("--skip_frame",type=int,default=3,help='skip frame of dataset')
    parser.add_argument("--pcd_sample",type=int,default=30000)
    parser.add_argument("--max_deg",type=float,default=10)  # 10deg in each axis  (see the paper)
    parser.add_argument("--max_tran",type=float,default=0.2)   # 0.2m in each axis  (see the paper)
    parser.add_argument("--mag_randomly",type=bool,default=True)
    # dataloader
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--num_workers",type=int,default=8)
    parser.add_argument("--pin_memory",type=bool,default=True,help='set it to False if your CPU memory is insufficient')
    # schedule
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument("--resume",type=str,default='')
    parser.add_argument("--pretrained",type=str,default='checkpoint/0.2-0.4-0.4(10,0.2)_input(gra&depth)_Ablation')
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--log_dir",default='log/')
    parser.add_argument("--checkpoint_dir",type=str,default="checkpoint/")
    parser.add_argument("--name",type=str,default='0.2-0.4-0.4(10,0.2)_input(gra&depth)_Ablation')
    parser.add_argument("--optim",type=str,default='adam',choices=['sgd','adam'])
    parser.add_argument("--lr0",type=float,default=1e-5)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--weight_decay",type=float,default=1e-4)
    parser.add_argument("--lr_exp_decay",type=float,default=0.98)
    parser.add_argument("--clip_grad",type=float,default=1.0)
    # setting
    parser.add_argument("--scale",type=float,default=50.0,help='scale factor of zhehzpcd normlization in loss')
    parser.add_argument("--inner_iter",type=int,default=1,help='inner iter of calibnet')
    parser.add_argument("--alpha",type=float,default=0.2,help='weight of gradient loss')
    parser.add_argument("--beta",type=float,default=0.4,help='weight of chamfer loss')
    parser.add_argument("--gama", type=float, default=0.4,help='weight of rt loss')
    parser.add_argument("--resize_ratio",type=float,nargs=2,default=[1.0,1.0])
    # if CUDA is out of memory, please reduce batch_size, pcd_sample or inner_iter
    return parser.parse_args()


@torch.no_grad()
def val(args,model:OTF_CalibNet,val_loader:DataLoader):
    model.eval()
    device = model.device
    tqdm_console = tqdm(total=len(val_loader),desc='Val')
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
    gradient_loss = loss_utils.GradientDifferenceLoss(scale=1.0, reduction='mean')
    rt_loss = loss_utils.RTLoss()
    alpha = float(args.alpha)
    beta = float(args.beta)
    gama = float(args.gama)
    total_dR = 0
    total_dT = 0
    total_loss = 0
    total_se3_loss = 0
    with tqdm_console:
        tqdm_console.set_description_str('Val')
        for batch in val_loader:
            rgb_img = batch['img'].to(device)
            B = rgb_img.size(0)
            pcd_range = batch['pcd_range'].to(device)
            calibed_depth_img = batch['depth_img'].to(device)
            calibed_pcd = batch['pcd'].to(device)
            uncalibed_pcd = batch['uncalibed_pcd'].to(device)
            uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
            InTran = batch['InTran'][0].to(device)
            igt = batch['igt'].to(device)
            img_shape = rgb_img.shape[-2:]
            depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
            # model(rgb_img,uncalibed_depth_img)
            g0 = torch.eye(4).repeat(B,1,1).to(device)
            for _ in range(args.inner_iter):
                rgb_gradient = compute_single_channel_gradients(rgb_img)  # 计算 RGB 图像的梯度
                # depth_gradient = compute_gradient(uncalibed_depth_img.unsqueeze(1))  # 计算深度图的梯度
                uncalibed_depth_img_gradient = compute_gradients(uncalibed_depth_img)
                # 在这里可以选择使用 rgb_gradient 或者 depth_gradient
                twist_rot, twist_tsl = model(rgb_gradient,uncalibed_depth_img)
                extran = utils.se3.exp(torch.cat([twist_rot,twist_tsl],dim=1))
                uncalibed_depth_img, uncalibed_pcd = depth_generator(extran,uncalibed_pcd)
                g0 = extran.bmm(g0)
            err_g = g0.bmm(igt)
            dR,dT = loss_utils.geodesic_distance(err_g)
            total_dR += dR.item()
            total_dT += dT.item()
            se3_loss = torch.linalg.norm(utils.se3.log(err_g),dim=1).mean()/6
            total_se3_loss += se3_loss.item()
            loss1 = photo_loss(calibed_depth_img,uncalibed_depth_img)
            loss2 = chamfer_loss(calibed_pcd,uncalibed_pcd)
            loss3 = (dR+dT)
            loss4 = (gradient_loss(calibed_depth_img,uncalibed_depth_img))
            loss = alpha*loss4 + loss2*beta +loss3*gama
            total_loss += loss.item()
            logger.info('dR:{:.4f}, dT:{:.4f},d_direct:{:.4f},d_gradient:{:.4f},d_loss_total:{:.4f},se3_loss:{:.4f}'.format(loss1,loss2,loss3,loss4,total_loss,se3_loss))
            tqdm_console.update(1)
    total_dR /= len(val_loader)
    total_dT /= len(val_loader)
    total_loss /= len(val_loader)
    total_se3_loss /= len(val_loader)
    tqdm_console.set_postfix_str(
        "total_dR:{:.4f}, total_dT:{:.4f},total_loss:{:.4f},total_se3_loss:{:.4f}".format(float(total_dR),
                                                                                        float(total_dT),
                                                                                        float(total_loss),
                                                                                        float(total_se3_loss),
                                                                                     ))

    tqdm_console.update(1)

    return total_loss, total_dR, total_dT, total_se3_loss

# 确保路径存在，如果不存在则创建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)# 确保路径存在，如果不存在则创建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def compute_gradients(depth_map):
    # 假设 depth_map 的形状是 (batch_size, 1, height, width)
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=depth_map.dtype, device=depth_map.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=depth_map.dtype, device=depth_map.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    # 计算水平和垂直梯度
    grad_x = F.conv2d(depth_map, sobel_x, padding=1)  # (batch_size, 1, height, width)
    grad_y = F.conv2d(depth_map, sobel_y, padding=1)  # (batch_size, 1, height, width)

    # 计算梯度幅值
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)  # (batch_size, 1, height, width)

    return  grad_magnitude

def compute_single_channel_gradients(rgb_batch):
    # 确保输入图像形状为 (batch_size, 3, height, width)
    device = rgb_batch.device  # 获取输入张量的设备
    if rgb_batch.ndim != 4 or rgb_batch.shape[1] != 3:
        raise ValueError("Input batch must have shape (batch_size, 3, height, width)")

    # Sobel算子的定义
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=rgb_batch.dtype).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 3)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=rgb_batch.dtype).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 3)

    gradients = []

    # 对每个通道进行卷积来计算梯度
    for i in range(3):  # 对R、G、B三个通道分别处理
        grad_x = (F.conv2d(rgb_batch[:, i:i + 1], sobel_x, padding=1)).to(device)  # (batch_size, 1, height, width)
        grad_y = (F.conv2d(rgb_batch[:, i:i + 1], sobel_y, padding=1)).to(device)  # (batch_size, 1, height, width)
        grad_magnitude = torch.sqrt(grad_x.squeeze() ** 2 + grad_y.squeeze() ** 2)  # 计算总梯度
        gradients.append(grad_magnitude.unsqueeze(1))  # 在第二维增加一维，变为 (batch_size, 1, height, width)

    # 合并所有通道的梯度图
    single_channel_gradient = torch.mean(torch.cat(gradients, dim=1), dim=1,
                                         keepdim=True)  # 取均值，输出形状为 (batch_size, 1, height, width)

    return single_channel_gradient


def train(args,chkpt,train_loader:DataLoader,val_loader:DataLoader):
    device = torch.device(args.device)
    model = OTF_CalibNet(backbone_pretrained=False,depth_scale=args.scale)
    model.to(device)
    # 创建文件夹
    depth_img_dir = 'depth_map'
    uncalib_depth_img_dir = './uncalib_depth_images'
    ensure_dir(depth_img_dir)
    ensure_dir(uncalib_depth_img_dir)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),args.lr0,momentum=args.momentum,weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(),args.lr0,weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_exp_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    #scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
    if args.pretrained:
        if os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            model.load_state_dict(torch.load(args.pretrained)['model'])
            print_highlight('Pretrained model loaded from {:s}'.format(args.pretrained))
        else:
            print_warning('Invalid pretrained path: {:s}'.format(args.pretrained))
    if chkpt is not None:
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        scheduler.load_state_dict(chkpt['scheduler'])
        start_epoch = chkpt['epoch'] + 1
        min_loss = chkpt['min_loss']
        log_mode = 'a'
    else:
        start_epoch = 0
        min_loss = float('inf')
        log_mode = 'w'
    if not torch.cuda.is_available():
        args.device = 'cpu'
        print_warning('CUDA is not available, use CPU to run')
    log_mode = 'a' if chkpt is not None else 'w'
    logger = get_logger("{name}|Train".format(name=args.name),os.path.join(args.log_dir,args.name+'.log'),mode=log_mode)
    if chkpt is None:
        logger.debug(args)
        print_highlight('Start Training')
    else:
        print_highlight('Resume from epoch {:d}'.format(start_epoch+1))
    del chkpt  # free memory
    photo_loss = loss_utils.Photo_Loss(args.scale)
    chamfer_loss = loss_utils.ChamferDistanceLoss(args.scale,'mean')
    rt_loss = loss_utils.RTLoss()
    gradient_loss = loss_utils.GradientDifferenceLoss(scale=1.0, reduction='mean')
    alpha = float(args.alpha)
    beta = float(args.beta)
    gama = float(args.gama)
    for epoch in range(start_epoch,args.epoch):
        model.train()
        tqdm_console = tqdm(total=len(train_loader),desc='Train')
        total_photo_loss = 0
        total_chamfer_loss = 0
        total_rt_loss=0
        total_gradient_loss=0
        has_nan = False  # 用于检查是否有NaN损失

        with tqdm_console:
            tqdm_console.set_description_str('Epoch: {:03d}|{:03d}'.format(epoch+1,args.epoch))
            for batch_idx,batch in enumerate(train_loader):
                optimizer.zero_grad()
                rgb_img = batch['img'].to(device)
                B = rgb_img.size(0)
                pcd_range = batch['pcd_range'].to(device)
                calibed_depth_img = batch['depth_img'].to(device)
                calibed_pcd = batch['pcd'].to(device)
                uncalibed_pcd = batch['uncalibed_pcd'].to(device)
                uncalibed_depth_img = batch['uncalibed_depth_img'].to(device)
                InTran = batch['InTran'][0].to(device)
                igt = batch['igt'].to(device)
                # 保存depth_img和uncalib_depth_img
                # for i in range(B):
                #     # 将张量移至CPU并转换为NumPy数组
                #     calibed_depth_img_2d = calibed_depth_img[i].cpu().numpy().squeeze()
                #     uncalibed_depth_img_2d = uncalibed_depth_img[i].cpu().numpy().squeeze()

                img_shape = rgb_img.shape[-2:]
                depth_generator = utils.transform.DepthImgGenerator(img_shape,InTran,pcd_range,CONFIG['dataset']['pooling'])
                # model(rgb_img,uncalibed_depth_img)
                Tcl = torch.eye(4).repeat(B,1,1).to(device)

                # model.eval()
                for _ in range(args.inner_iter):
                    print(rgb_img.shape)
                    rgb_gradient = compute_single_channel_gradients(rgb_img)  # 计算 RGB 图像的梯度
                    print(rgb_gradient.shape)
                    #depth_gradient = compute_gradient(uncalibed_depth_img.unsqueeze(1))  # 计算深度图的梯度
                    # 在这里可以选择使用 rgb_gradient 或者 depth_gradient
                    uncalibed_depth_img_gradient=compute_gradients(uncalibed_depth_img)
                    twist_rot, twist_tsl = model(rgb_gradient, uncalibed_depth_img)
                    iter_Tcl = utils.se3.exp(torch.cat([twist_rot, twist_tsl], dim=1))
                    uncalibed_depth_img, uncalibed_pcd = depth_generator(iter_Tcl, uncalibed_pcd)
                    Tcl = Tcl.bmm(iter_Tcl)

                dR,dT = (loss_utils.geodesic_distance(Tcl.bmm(igt)))
                # model.train()
                loss1 = (photo_loss(calibed_depth_img,uncalibed_depth_img))
                loss4 = (gradient_loss(calibed_depth_img,uncalibed_depth_img))
                loss2 = (chamfer_loss(calibed_pcd,uncalibed_pcd))
                loss3 = (dR + dT)
                loss =  alpha*loss4 + beta*loss2+ gama*loss3
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(),args.clip_grad)
                optimizer.step()
                tqdm_console.set_postfix_str("p:{:.4f}, c:{:.4f},direct:{:.4f},gradient:{:.4f}, dR:{:.4f}, dT:{:.4f}".format(float(loss1.item()),float(loss2.item()),float(loss3.item()),float(loss4.item()),float(dR.item()),float(dT.item())))
                tqdm_console.update()
                total_photo_loss += loss1.item()
                total_chamfer_loss += loss2.item()
                total_rt_loss += loss3.item()
                total_gradient_loss += loss4.item()
        # N_loader = len(train_loader)
        # total_photo_loss /= N_loader
        # total_chamfer_loss /= N_loader
        total_loss =  (alpha*total_gradient_loss + beta*total_chamfer_loss + gama*total_rt_loss)/len(train_loader)
        tqdm_console.set_postfix_str("loss: {:.3f}, photo: {:.3f}, chamfer: {:.3f}".format(float(total_loss),float(total_photo_loss),float(total_chamfer_loss)))
        tqdm_console.update()
        tqdm_console.close()
        logger.info('Epoch {:03d}|{:03d}, train loss:{:.4f}'.format(epoch+1,args.epoch,total_loss))
        scheduler.step()
        val_loss, loss_dR, loss_dT, loss_se3 = val(args,model,val_loader)  # float 
        if loss_se3 < min_loss:
            min_loss = loss_se3
            torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_best.pth'.format(name=args.name)))
            logger.debug('Best model saved (Epoch {:d})'.format(epoch+1))
            print_highlight('Best Model (Epoch %d)'%(epoch+1))
            logger.info("val_loss: {:.6f}".format(val_loss))
            logger.info("loss_se3: {:.6f}".format(loss_se3))

        torch.save(dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                min_loss=min_loss,
                epoch=epoch,
                args=args.__dict__,
                config=CONFIG
            ),os.path.join(args.checkpoint_dir,'{name}_last.pth'.format(name=args.name)))
        logger.info('Evaluate loss_dR:{:.6f}, loss_dT:{:.6f}, se3_loss:{:.6f}'.format(loss_dR,loss_dT,loss_se3))
            
            
            

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')  # 使用 GPU
    else:
        device = torch.device('cpu')  # 使用 CPU
    args = options()
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.checkpoint_dir,exist_ok=True)
    with open(args.config,'r')as f:
        CONFIG = yaml.load(f,yaml.SafeLoader)
    assert isinstance(CONFIG,dict), 'Unknown config format!'
    if args.resume:
        chkpt = torch.load(args.resume,map_location='cpu')
        CONFIG.update(chkpt['config'])
        args.__dict__.update(chkpt['args'])
        print_highlight('config updated from resumed checkpoint {:s}'.format(args.resume))
    else:
        chkpt = None
    print_highlight('args have been received, please wait for dataloader...')
    train_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['train']]
    val_split = [str(index).rjust(2,'0') for index in CONFIG['dataset']['val']]
    # dataset
    train_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,train_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=args.skip_frame,voxel_size=CONFIG['dataset']['voxel_size'],
                                     pcd_sample_num=args.pcd_sample,resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG['dataset']['extend_ratio'])
    train_dataset = KITTI_perturb(train_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                  pooling_size=CONFIG['dataset']['pooling'])
    
    val_dataset = BaseKITTIDataset(args.dataset_path,args.batch_size,val_split,CONFIG['dataset']['cam_id'],
                                     skip_frame=args.skip_frame,voxel_size=CONFIG['dataset']['voxel_size'],
                                     pcd_sample_num=args.pcd_sample,resize_ratio=args.resize_ratio,
                                     extend_ratio=CONFIG['dataset']['extend_ratio'])

    val_perturb_file = os.path.join(args.checkpoint_dir,"val_seq.csv")

    train_length = len(train_dataset)
    val_length = len(val_dataset)
    print(train_length,val_length)

    if not os.path.exists(val_perturb_file):
        print_highlight("validation pertub file dosen't exist, create one.")
        transform = UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
        perturb_arr = np.zeros([val_length,6])
        for i in range(val_length):
            perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
        np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
    else:  # check length
        val_seq = np.loadtxt(val_perturb_file,delimiter=',')
        if val_length != val_seq.shape[0]:
            print_warning('Incompatiable validation length {}!={}'.format(val_length,val_seq.shape[0]))
            transform = utils.transform.UniformTransformSE3(args.max_deg,args.max_tran,args.mag_randomly)
            perturb_arr = np.zeros([val_length,6])
            for i in range(val_length):
                perturb_arr[i,:] = transform.generate_transform().cpu().numpy()
            np.savetxt(val_perturb_file,perturb_arr,delimiter=',')
            print_highlight('Validation perturb file rewritten.')

    val_dataset = KITTI_perturb(val_dataset,args.max_deg,args.max_tran,args.mag_randomly,
                                pooling_size=CONFIG['dataset']['pooling'],
                                file=os.path.join(args.checkpoint_dir,"val_seq.csv"))

    # batch normlization does not support batch=1
    train_drop_last = True if len(train_dataset) % args.batch_size == 1 else False  
    val_drop_last = True if len(val_dataset) % args.batch_size == 1 else False

    # dataloader
    train_dataloader = DataLoader(train_dataset,args.batch_size,shuffle=False,num_workers=4,pin_memory=args.pin_memory,drop_last=train_drop_last)
    val_dataloder = DataLoader(val_dataset,args.batch_size,shuffle=False,num_workers=4,pin_memory=args.pin_memory,drop_last=val_drop_last)
    
        
    train(args,chkpt,train_dataloader,val_dataloder)