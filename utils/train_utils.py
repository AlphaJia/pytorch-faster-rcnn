import datetime
import pickle
import time
from collections import defaultdict, deque

import torch.distributed as dist
from torchvision import ops

from backbone.mobilenet import MobileNetV2
from backbone.resnet50_fpn_model import *
from config.train_config import cfg
from utils.anchor_utils import AnchorsGenerator
from utils.faster_rcnn_utils import FasterRCNN, FastRCNNPredictor


def create_model(num_classes):
    global backbone, model
    backbone_network = cfg.backbone

    anchor_sizes = tuple((f,) for f in cfg.anchor_size)
    aspect_ratios = tuple((f,) for f in cfg.anchor_ratio) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    if backbone_network == 'mobilenet':
        backbone = MobileNetV2(weights_path=cfg.backbone_pretrained_weights).features
        backbone.out_channels = 1280

        roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],  # roi pooling in which resolution feature
                                            output_size=cfg.roi_out_size,  # roi_pooling output feature size
                                            sampling_ratio=cfg.roi_sample_rate)  # sampling_ratio

        model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                           # transform parameters
                           min_size=cfg.min_size, max_size=cfg.max_size,
                           image_mean=cfg.image_mean, image_std=cfg.image_std,
                           # rpn parameters
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                           rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
                           rpn_nms_thresh=cfg.rpn_nms_thresh,
                           rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
                           rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
                           rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
                           rpn_positive_fraction=cfg.rpn_positive_fraction,
                           # Box parameters
                           box_head=None, box_predictor=None,

                           # remove low threshold target
                           box_score_thresh=cfg.box_score_thresh,
                           box_nms_thresh=cfg.box_nms_thresh,
                           box_detections_per_img=cfg.box_detections_per_img,
                           box_fg_iou_thresh=cfg.box_fg_iou_thresh,
                           box_bg_iou_thresh=cfg.box_bg_iou_thresh,
                           box_batch_size_per_image=cfg.box_batch_size_per_image,
                           box_positive_fraction=cfg.box_positive_fraction,
                           bbox_reg_weights=cfg.bbox_reg_weights
                           )
    elif backbone_network == 'resnet50_fpn':
        backbone = resnet50_fpn_backbone()

        roi_pooler = ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=cfg.roi_out_size,
            sampling_ratio=cfg.roi_sample_rate)
        model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                           # transform parameters
                           min_size=cfg.min_size, max_size=cfg.max_size,
                           image_mean=cfg.image_mean, image_std=cfg.image_std,
                           # rpn parameters
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                           rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
                           rpn_nms_thresh=cfg.rpn_nms_thresh,
                           rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
                           rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
                           rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
                           rpn_positive_fraction=cfg.rpn_positive_fraction,
                           # Box parameters
                           box_head=None, box_predictor=None,

                           # remove low threshold target
                           box_score_thresh=cfg.box_score_thresh,
                           box_nms_thresh=cfg.box_nms_thresh,
                           box_detections_per_img=cfg.box_detections_per_img,
                           box_fg_iou_thresh=cfg.box_fg_iou_thresh,
                           box_bg_iou_thresh=cfg.box_bg_iou_thresh,
                           box_batch_size_per_image=cfg.box_batch_size_per_image,
                           box_positive_fraction=cfg.box_positive_fraction,
                           bbox_reg_weights=cfg.bbox_reg_weights
                           )

        # weights_dict = torch.load(cfg.pretrained_weights)
        # missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        # if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        #     print("missing_keys: ", missing_keys)
        #     print("unexpected_keys: ", unexpected_keys)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,

                                                         total_time / len(iterable)))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq,
                    train_loss=None, train_lr=None, warmup=False):
    global loss_dict, losses
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if isinstance(train_loss, list):
            train_loss.append(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        if isinstance(train_lr, list):
            train_lr.append(now_lr)

    return loss_dict, losses


def write_tb(writer, num, info):
    for item in info.items():
        writer.add_scalar(item[0], item[1], num)
