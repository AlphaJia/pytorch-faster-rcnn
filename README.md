<table border="0" width="10%">
  <tr>
    <td><img src="https://img1.github.io/tmp/1.jpg" height="80" width="82"></td>
    <td><img src="https://img1.github.io/tmp/2.jpg" height="80" width="82"></td>
    <td><img src="https://img1.github.io/tmp/3.jpg" height="80" width="82"></td>
  </tr>
  <tr>
    <td><img src="https://img1.github.io/tmp/4.jpg" height="80" width="82"></td>
    <td><img src="https://img.shields.io/github/stars/AlphaJia/pytorch-faster-rcnn.svg?style=social"></td>
    <td><img src="https://img1.github.io/tmp/6.jpg" height="82" width="82"></td>
  </tr>
   <tr>
    <td><img src="https://img1.github.io/tmp/7.jpg" height="82" width="82"></td>
    <td><img src="https://img1.github.io/tmp/8.jpg" height="82" width="82"></td>
    <td><img src="https://img1.github.io/tmp/9.jpg" height="82" width="82"></td>
  </tr>
</table>

# pytorch-faster-rcnn
* [pytorch\-faster\-rcnn](#pytorch-faster-rcnn)
    * [Framework Structure](#framework-structure)
      * [backbone](#backbone)
      * [config](#config)
      * [dataloader](#dataloader)
      * [test](#test)
      * [utils](#utils)  

pytorch based implementation of faster rcnn framework([Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497))

### Framework Structure  
#### backbone
This module includes backbone feature extraction network    
* vgg16:vgg16 net network([Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556))
* fpn101:resnet101 fpn network([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)) ([Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144))
* hrnet:high resolution net([Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919))
* mobile_net:mobile_net v2 network([MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381))
#### config
This module includes config parameters in training period  and testing period
* test_config: specify config parameters in testing period like model_file, image_path_dir, save_dir, etc.
* train_config: specify config parameters in training period like backbone network, batch_size, image_path_dir, anchor_size, ect.
#### dataloader
This module inherits pytorch dataloader classes, dataset IO.You can also generate your own dataset dataloader IO and put it in this module
* coco_dataset: coco([Common Objects in Context](https://cocodataset.org/#home)) dataset dataloader IO
#### test
This module includes the utils function test(common called unit test, also called UT)
* anchor_utils_test: some unit testing for utils/anchor_utils.py
#### utils
This module includes some utilies for image processing, network architectures building, anchor generating, loss function, etc.
* anchor_utils: some basic function for building anchors
* im_utils: some basic function for image processing

# Continue Updating
