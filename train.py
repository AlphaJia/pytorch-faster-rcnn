import os

import torch
from tensorboardX import SummaryWriter

from config.train_config import cfg
from dataloader.coco_dataset import coco
from utils.evaluate_utils import evaluate
from utils.im_utils import Compose, ToTensor, RandomHorizontalFlip
from utils.plot_utils import plot_loss_and_lr, plot_map
from utils.train_utils import train_one_epoch, write_tb, create_model


def main():
    device = torch.device(cfg.device_name)
    print("Using {} device training.".format(device.type))

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(cfg.model_save_dir, 'epoch_log'))

    data_transform = {
        "train": Compose([ToTensor(), RandomHorizontalFlip(cfg.train_horizon_flip_prob)]),
        "val": Compose([ToTensor()])
    }

    if not os.path.exists(cfg.data_root_dir):
        raise FileNotFoundError("dataset root dir not exist!")

    # load train data set
    train_data_set = coco(cfg.data_root_dir, 'train', '2017', data_transform["train"])
    batch_size = cfg.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=nw,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    val_data_set = coco(cfg.data_root_dir, 'val', '2017', data_transform["val"])
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=train_data_set.collate_fn)

    # create model num_classes equal background + 80 classes
    model = create_model(num_classes=cfg.num_class)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.lr_dec_step_size,
                                                   gamma=cfg.lr_gamma)

    # train from pretrained weights
    if cfg.resume != "":
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(cfg.start_epoch))

    train_loss = []
    learning_rate = []
    train_mAP_list = []
    val_mAP = []

    best_mAP = 0
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        loss_dict, total_loss = train_one_epoch(model, optimizer, train_data_loader,
                                                device, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=50, warmup=False)

        lr_scheduler.step()

        print("------>Starting training data valid")
        _, train_mAP = evaluate(model, train_data_loader, device=device, mAP_list=train_mAP_list)

        print("------>Starting validation data valid")
        _, mAP = evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)
        print('training mAp is {}'.format(train_mAP))
        print('validation mAp is {}'.format(mAP))
        print('best mAp is {}'.format(best_mAP))

        board_info = {'lr': optimizer.param_groups[0]['lr'],
                      'train_mAP': train_mAP,
                      'val_mAP': mAP}

        for k, v in loss_dict.items():
            board_info[k] = v.item()
        board_info['total loss'] = total_loss.item()
        write_tb(writer, epoch, board_info)

        if mAP > best_mAP:
            best_mAP = mAP
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            model_save_dir = cfg.model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(save_files,
                       os.path.join(model_save_dir, "{}-model-{}-mAp-{}.pth".format(cfg.backbone, epoch, mAP)))
    writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, cfg.model_save_dir)

    # plot mAP curve
    if len(val_mAP) != 0:
        plot_map(val_mAP, cfg.model_save_dir)


if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    main()
