# pytorch-faster-rcnn
pytorch based implementation of faster rcnn([Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497))

目录结构  
backbone:特征提取主干网络      
|________________vgg16  
|________________fpn101  
|________________mobile_net(#TODO)

utilies:训练及预测过程中得功能性函数  

dataloader:dataset函数，数据预处理