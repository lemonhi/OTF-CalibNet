
# OTF-CalibNet: On-the-fly Calibration Network for Synchronous LiDAR-Camera Systems

the code was implemented by Jiahui Chai for the paper OTF-CalibNet
## Implemented Environment

Ubuntu 20.04

Pytorch 2.4.0

CUDA 12.4

Python 3.12.5

Intel i9 13900K CPU and NVIDIA GeForce RTX4090 GPU

## Dataset Preparation

KITTI Odometry 
http://www.cvlibs.net/datasets/kitti/eval_odometry.php
Dataset Should be organized into data/ filefolder in our root:
/PATH/TO/OTF-CalibNet/
  --|data/
      --|poses/
          --|00.txt
          --|01.txt
          --...
      --|sequences/
          --|00/
              --|image_2/
              --|image_3/
              --|velodyne/
              --|calib.txt
              --|times.txt
          --|01/
          --|02/
          --...
  --...
## Train and Test
### Train

The following command is fit with a 24GB GPU.

```
python train.py or 直接点击运行train.py文件即可

```
### Test

```
python test.py or 直接点击运行test.py文件即可
```
Download pretrained `链接: https://pan.baidu.com/s/1YwY65gI1ywYRKmA1L_we4A?pwd=sk8u 提取码: sk8u` from here and put it into root/checkpoint/.
## Results on KITTI Odometry Test (Seq = 11, one iter)
![](https://i-blog.csdnimg.cn/direct/bdbcb721534d426db180df9b14553c76.jpeg#pic_center)左上角为原始图像，右上角为gt外参进行可视化结果
右下角为受扰动外参可视化结果，右下角为通过OTF-CalibNet网络标定后的可视化结果

初始误差（±20°，±1.5m），去扰动标定后：
Rotation (deg)： X:1.0896,		Y:3.7543,		Z:1.3395
Translation (m)：X:0.03298,	Y:0.03586,		Z:0.03267

初始误差（±10°，±0.2m），去扰动标定后：
Rotation (deg)： X:0.159,		Y:0.328,		Z:0.315
Translation (m)：X:0.0231,	Y:0.0224,		Z:0.0206

初始误差（±2°，±0.2m），去扰动标定后：
Rotation (deg)： X:0.17,		Y:0.27,		Z:0.12
Translation (m)： X:0.025,	Y:0.0148,		Z:0.0320

## Notice

 1. 训练集需要提前自行下载； 
 2. 在test中有可视化的相关部分，为了确保代码能顺利运行，新建部分文件夹即可；

