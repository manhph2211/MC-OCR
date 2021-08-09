import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.backbone import ResNetBackBone
from utils.box_utils import match, crop, make_anchors


class FPN(nn.Module):
    def __init__(self, inchannels=(512, 1024, 2048)):
        super(FPN, self).__init__()
        self.inchannels = inchannels
        self.num_lateral_layers = len(inchannels)
        self.lateral_layers = nn.ModuleList()
        self.pred_layers = nn.ModuleList()

        for channel in reversed(self.inchannels):
            self.lateral_layers.append(nn.Conv2d(channel, 256, kernel_size=1))
            self.pred_layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))

        self.downsample_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downsample_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, resnet_output):
        outs = []

        p5_1 = self.lateral_layers[0](resnet_output[2])
        total = torch.zeros(p5_1.shape, device=resnet_output[1].device)
        total += p5_1

        _, _, h, w = resnet_output[1].size()
        total = F.interpolate(total, size=(h, w), mode="bilinear", align_corners=False)
        p4_1 = self.lateral_layers[1](resnet_output[1])
        total += p4_1
        p4_1 = total

        _, _, h, w = resnet_output[0].size()
        total = F.interpolate(total, size=(h, w), mode="bilinear", align_corners=False)
        p3_1 = self.lateral_layers[2](resnet_output[0])
        total += p3_1
        p3_1 = total

        p5 = F.relu(self.pred_layers[2](p5_1))
        p4 = F.relu(self.pred_layers[1](p4_1))
        p3 = F.relu(self.pred_layers[0](p3_1))

        outs.append(p3)
        outs.append(p4)
        outs.append(p5)

        outs.append(self.downsample_1(outs[-1]))  # outs append p6
        outs.append(self.downsample_2(outs[-1]))  # outs append p7

        return outs  # outs: [p3, p4, p5, p6, p7]


class Protonet(nn.Module):
    def __init__(self, coefdim=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)))
        self.proto_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.proto_conv2 = nn.Conv2d(256, coefdim, kernel_size=1)

    def forward(self, x):
        out = x  # (B, 256, 69, 69)
        for layer in self.layers:
            out = layer(out)
        out = F.interpolate(out, (138, 138), mode='bilinear', align_corners=False)  # (B, 256, 138, 138)
        out = self.relu(out)
        out = self.proto_conv1(out)  # (B, 256, 138, 138)
        out = self.relu(out)
        out = self.proto_conv2(out)  # (B, 32, 138, 138)
        return out


class Prediction_module(nn.Module):
    def __init__(self, cfg, coefdim=32):
        super(Prediction_module, self).__init__()
        self.num_classes = cfg.num_classes
        self.coefdim = coefdim
        self.num_anchors = len(cfg.aspect_ratios)

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bbox_layer = nn.Conv2d(256, self.num_anchors * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(256, self.num_anchors * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(256, self.num_anchors * self.coefdim, kernel_size=3, padding=1)

    def forward(self, x):
        # x is the tensor with the shape of (B, C, H, W)
        x = self.conv1(x)
        x = self.relu(x)

        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)  # (B, HxW, C)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)  # (B, HxW, 4)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coefdim)  # (B, HxW, 32)

        coef = torch.tanh(coef)

        return conf, box, coef


class Yolact(nn.Module):
    def __init__(self, cfg):
        super(Yolact, self).__init__()
        self.cfg = cfg
        self.anchors = []

        self.backbone = ResNetBackBone()  # Construct Resnet back bone
        self.fpn = FPN([512, 1024, 2048])  # Construct FPN
        self.protonet = Protonet()

        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(Prediction_module(cfg, coefdim=32))  # Construct Prediction Module

        if cfg.mode == "train":
            # Use the semantic segmentation convoluton layer only in training mode
            channels_out = cfg.num_classes - 1
            self.semantic_seg_conv = nn.Conv2d(256, channels_out, kernel_size=1)

    def load_weights(self, path):
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location="cpu")

        for key in list(state_dict.keys()):
            # 'fpn.downsample_layers.2.weight' and 'fpn.downsample_layers.2.bias'
            # in the pretrained .pth are redundant, remove them
            if key.startswith('fpn.downsample.'):
                if int(key.split('.')[2]) >= 2:
                    del state_dict[key]

            if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self):
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone()
        # Initialize the rest conv layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and not name.startswith('backbone'):
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, img, box_classes=None, mask_gt=None):
        # Pass through Resnet and FPN
        with torch.no_grad():
            outs = self.backbone(img)
        outs = self.fpn(outs)  # [P3, P4, P5, P6, P7]

        # specify anchor boxes in predicted images
        if isinstance(self.anchors, list):
            for i, shape in enumerate([list(aa.shape) for aa in outs]):
                self.anchors += make_anchors(self.cfg, shape[2], shape[3], self.cfg.scales[i])

        self.anchors = torch.tensor(self.anchors, device=outs[0].device).reshape(-1, 4)

        # Pass FPN P3 through protonet to generate prototypes
        proto_out = self.protonet(outs[0])
        proto_out = F.relu(proto_out, inplace=True)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        class_pred, box_pred, coef_pred = [], [], []
        for out in outs:
            # Pass each out put of FPN to get anchor boxes prediction
            class_p, box_p, coef_p = self.prediction_layers[0](out)
            class_pred.append(class_p)
            box_pred.append(box_p)
            coef_pred.append(coef_p)

        # Concat all the predictions to get shape (19248, _type)
        class_pred = torch.cat(class_pred, dim=1)  # (19248, num_class)
        box_pred = torch.cat(box_pred, dim=1)  # (19248, 4)
        coef_pred = torch.cat(coef_pred, dim=1)  # (19248, 32)

        if self.training:
            seg_pred = self.semantic_seg_conv(outs[0])
            return self.compute_loss(class_pred, box_pred, coef_pred, proto_out, seg_pred, box_classes, mask_gt)
        else:
            class_pred = F.softmax(class_pred, -1)
            return class_pred, box_pred, coef_pred, proto_out, self.anchors

    def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        num_anchors = self.anchors.shape[0]

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)

        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre'
        # 0 means background, '>0' means foreground.
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)

        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        # create training ground truth data
        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]  # first 4 values contain coordinates
            class_gt[i] = box_class[i][:, -1].long()  # last value contains ground truth class
            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] \
                = match(self.cfg, box_gt, self.anchors, class_gt[i])  # Match corresponding anchor box

        # only compute losses from positive samples
        pos_bool = conf_gt > 0  # (n, 19248)
        
        loss_c = self.category_loss(class_p, conf_gt, pos_bool)
        loss_b = self.box_loss(box_p, all_offsets, pos_bool)
        loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)
        loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)

        return loss_c, loss_b, loss_m, loss_s

    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)
        batch_conf_max = batch_conf.max()  # subtract max for numerical stability
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        mark = mark.reshape(class_p.size(0), -1)  # (n, 19248)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals
        
        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]
        
        return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined,
                                                     reduction='sum') / num_pos.sum()  # cross entropy loss

    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]  # use only positive samples
        pos_offsets = all_offsets[pos_bool, :]

        return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets,
                                                      reduction='sum') / num_pos  # smooth l1 regression box loss

    def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
        for i in range(coef_p.size(0)):  # coef_p.shape: (n, 19248, 32)
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2,
                                                          0).contiguous()  # (138, 138, num_objects) every ground truth object has one mask
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:  # if no positive samples , continue
                continue
            # If number of positive anchors exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]

            num_pos = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            # mask assembly by linear combination ,  @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())  # mask_p.shape: (138, 138, num_pos)
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
            # aa = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)
            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)

        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # classes here exclude the background class, so num_classes = cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()  # (n, 20, 69, 69)
        loss_s = 0
        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()  # (num_objects, 69, 69)

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])
            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size
