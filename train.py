import os
import torch
from torchvision import ops
from config.train_config import cfg
from dataloader.coco_dataset import coco
from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip
from utils.anchor_utils import AnchorsGenerator
from backbone.mobilenet import MobileNetV2
from utils.faster_rcnn_utils import FasterRCNN


def create_model(num_classes):
    backbone_network = cfg.backbone
    if backbone_network == 'mobilenet':
        backbone = MobileNetV2(weights_path=cfg.pretrained_weights).features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=cfg.anchor_size,
                                        aspect_ratios=cfg.anchor_ratio)

    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                        output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                        sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main():
    device = torch.device(cfg.device_name)
    print("Using {} device training.".format(device.type))

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    data_transform = {
        "train": Compose([ToTensor(), RandomHorizontalFlip(cfg.train_horizon_flip_prob)]),
        "val": Compose([ToTensor()])
    }

    if not os.path.exists(cfg.data_root_dir):
        raise FileNotFoundError("dataset root dir not exist!")

    # load train data set
    train_data_set = coco(cfg.data_root_dir, 'train', '2017', data_transform["train"])
    batch_size = cfg.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    val_data_set = coco(cfg.data_root_dir, 'val', '2017', data_transform["train"])
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=cfg.num_class)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if cfg.resume != "":
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_mAP = []

    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        # train for one epoch, printing every 10 iterations
        utils.train_one_epoch(model, optimizer, train_data_loader,
                              device, epoch, train_loss=train_loss, train_lr=learning_rate,
                              print_freq=50, warmup=True)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        utils.evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        model_save_dir = cfg['model_save_dir']
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        torch.save(save_files, os.path.join(model_save_dir, "{}-model-{}.pth".format(cfg['backbone'], epoch)))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from plot_curve import plot_map
        plot_map(val_mAP)

    # model.eval()
    # x = [torch.rand(3, 300, 400), torch.rand(3, 400, 400)]
    # predictions = model(x)
    # print(predictions)


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    main()
