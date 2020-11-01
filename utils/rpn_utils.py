from torch.nn import functional as F
from torch import nn
from utils.det_utils import *
from torch.jit.annotations import List, Optional, Dict, Tuple
import utils.boxes_utils as box_op


class RPNHead(nn.Module):
    """
     RPN head with background/foreground classification and bbox regression
     :param self:
     :param in_channels: number of channels of the input feature
     :param num_anchors: number of anchors to be predicted
     :return:
    """

    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()
        # 3x3 conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # background/foreground score
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # bbox regression parameters
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        cls_scores = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            cls_scores.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return cls_scores, bbox_reg


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
        'fg_bg_sampler': BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_op.box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_op.box_iou(gt_boxes, anchors_per_image)
                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数）
        Returns:

        """
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):

            num_anchors = ob.shape[1]  # 预测特征层上的预测的anchors个数
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)  # self.pre_nms_top_n=1000

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        Args:
            proposals: 预测的bbox坐标
            objectness: 预测的目标概率
            image_shapes: batch中每张图片的size信息
            num_anchors_per_level: 每个预测特征层上预测anchors的数目

        Returns:

        """
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # 获取每张预测特征图上预测概率排前pre_nms_top_n的anchors索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        # 根据每个预测特征层预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_op.clip_boxes_to_image(boxes, img_shape)
            # 返回boxes满足宽，高都大于min_size的索引
            keep = box_op.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # non-maximum suppression, independently done per level
            keep = box_op.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率
            pred_bbox_deltas (Tensor)：预测的bbox regression
            labels (List[Tensor])：真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
            regression_targets (List[Tensor])：真实的bbox regression

        Returns:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失
        box_loss = smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features是所有预测特征层组成的OrderedDict
        features = list(features.values())

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list
        objectness, pred_bbox_deltas = self.head(features)

        # 生成一个batch图像的所有anchors信息,list(tensor)元素个数等于batch_size
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        # 计算每个预测特征层上的对应的anchors数量
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整内部tensor格式以及shape
        objectness, pred_bbox_deltas = box_op.concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        # 将预测的bbox regression参数应用到anchors上得到最终预测bbox坐标
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses
