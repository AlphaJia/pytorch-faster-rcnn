from torch import nn
from torch.jit.annotations import Dict
from torch.nn import functional as F

import utils.boxes_utils as box_op
from utils.det_utils import *


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
     Implementation of Region Proposal Network (RPN).
     :param anchor_generator: module that generates the anchors for feature map.
     :param head: module that computes the objectness and regression deltas
     :param fg_iou_thresh: minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
     :param bg_iou_thresh: maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
     :param batch_size_per_image: number of anchors that are sampled during training of the RPN
            for computing the loss
     :param positive_fraction: proportion of positive anchors in a mini-batch during training
            of the RPN
     :param pre_nms_top_n: number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
     :param post_nms_top_n: number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
     :param nms_thresh: NMS threshold used for postprocessing the RPN proposals
     """

    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):

        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # function for computing iou between anchor and true bbox
        self.box_similarity = box_op.box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # foreground threshold, if IOU > threshold(0.7), is positive samples
            bg_iou_thresh,  # background threshold, if IOU < threshold(0.3), is negative samples
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
        """
        get the best match gt for anchors, divided into bg samples, fg samples and unused samples
        :param anchors: (List[Tensor])
        :param targets: (List[Dict[Tensor])
        :return: labels: anchors cls, 1 is foreground, 0 is background, -1 is unused
            matched_gt_boxes：best matched gt
        """

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # compute iou of anchors and real bbox
                match_quality_matrix = box_op.box_iou(gt_boxes, anchors_per_image)
                # calculate index of anchors and gt iou（iou<0.3 is -1，0.3<iou<0.7 is -2）
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
        get thr top pre_nms_top_n anchor index in predicted feature_maps based on scores
        :param objectness: scores
        :param num_anchors_per_level: number of anchors
        :return:
        """

        result = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(result, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        """
        remove small bboxes, nms process, get post_nms_top_n target
        :param proposals: predicted bbox coordinates
        :param objectness: predicted scores
        :param image_shapes: image shape
        :param num_anchors_per_level: number od anchors of per feature_maps
        :return:
        """

        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size, size filled with fill_value
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            # adjust predicted bbox, make boxes outside of the image in image
            boxes = box_op.clip_boxes_to_image(boxes, img_shape)

            # Remove boxes which contains at least one side smaller than min_size.
            keep = box_op.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_op.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only top k scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        """
        compute RPN loss, include classification loss(foreground and background), bbox regression loss
        :param objectness: predicted foreground probability
        :param pred_bbox_deltas: predicted bbox regression parameters
        :param labels: true lable, 1, 0 and -1
        :param regression_targets: true bbox regression
        :return: objectness_loss (Tensor) : classification loss
                 box_loss (Tensor)：bbox loss
        """

        # selective positive and negative samples
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # bbox regression loss
        box_loss = smooth_l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds],
                                  beta=1 / 9, size_average=False, ) / (sampled_inds.numel())

        # classification loss
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        """
        :param images: (ImageList), images for which we want to compute the predictions
        :param features: (Dict[Tensor]), features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
        :param targets: (List[Dict[Tensor]), ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        :return:
              boxes (List[Tensor]): the predicted boxes from the RPN image.
              losses (Dict[Tensor]): the losses for the model during training. During testing, it is an empty dict.
        """

        # RPN uses all feature maps that are available
        features = list(features.values())

        # Two fc layers to compute the fg/bg scores and bboxs regressions
        fg_bg_scores, pred_bbox_deltas = self.head(features)

        # get all anchors of images based on features
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        num_anchors_per_level_shape_tensors = [o[0].shape for o in fg_bg_scores]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # adjust tensor order and reshape
        fg_bg_scores, pred_bbox_deltas = box_op.concat_box_prediction_layers(fg_bg_scores, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # remove small bboxes, nms process, get post_nms_top_n target
        boxes, scores = self.filter_proposals(proposals, fg_bg_scores, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # encode parameters based on the bboxes and anchors
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                fg_bg_scores, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}

        return boxes, losses
