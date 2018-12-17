import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import convert_to_one_hot, get_mean_size_array, get_box3d_corners, get_box3d_corners_helper
import numpy as np
from torch.autograd import Variable


class Frustum3DLoss(nn.Module):
    def __init__(self, num_heading_bin, num_size_cluster, endpoints, device='cuda',
                 corner_loss_weight=10.0, box_loss_weight=1.0):
        super(Frustum3DLoss, self).__init__()
        self.device = device
        self.corner_loss_weight = corner_loss_weight
        self.box_loss_weight = box_loss_weight
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.one_hot_hc_label = None
        self.one_hot_sc_label = None
        self.mean_size_array = torch.Tensor(get_mean_size_array(self.num_size_cluster)).to(self.device)
        self.losses = {}
        self.endpoints = endpoints

    def forward(self, mask_label,
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label):
        endpoints = self.endpoints
        seg_loss = self.get_segmentation_loss(mask_label, endpoints['mask_logits'])
        self.losses['seg_loss'] = seg_loss

        center_loss, stage1_center_loss = self.get_center_losses(center_label, endpoints['center'],
                                                                 endpoints['stage1_center'])
        self.losses['center_loss'] = center_loss
        self.losses['stage1_center_loss'] = stage1_center_loss

        heading_class_loss, heading_residual_normalized_loss, self.one_hot_hc_label = self.get_heading_loss(
            heading_class_label, endpoints['heading_scores'], heading_residual_label,
            endpoints['heading_residuals_normalized'])
        self.losses['heading_class_loss'] = heading_class_loss
        self.losses['heading_residual_normalized_loss'] = heading_residual_normalized_loss

        size_class_loss, size_residuals_normalized_loss, self.one_hot_sc_label = self.get_size_loss(
            endpoints['size_scores'], size_class_label, size_residual_label,
            endpoints['size_residuals_normalized']
        )
        self.losses['size_class_loss'] = size_class_loss
        self.losses['size_residuals_normalized_loss'] = size_residuals_normalized_loss

        corner_loss = self.get_corner_loss(center_label, endpoints['center'],
                                           heading_residual_label, endpoints['heading_residuals'],
                                           size_residual_label, endpoints['size_residuals'])
        self.losses['corner_loss'] = corner_loss

        total_loss = seg_loss + self.box_loss_weight * (center_loss +heading_class_loss + size_class_loss +
                                                        heading_residual_normalized_loss * 20 +
                                                        size_residuals_normalized_loss * 20 +
                                                        stage1_center_loss +
                                                        self.corner_loss_weight * corner_loss)
        self.losses['total_loss'] = total_loss

        return total_loss

    def huber_loss(self, error, delta):
        abs_error = torch.abs(error)
        delta = torch.Tensor([delta]).to(self.device)
        quadratic = torch.min(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(losses)

    def get_segmentation_loss(self, mask_label, segmentation_logits):
        return F.cross_entropy(segmentation_logits.permute(0, 2, 1),
                               mask_label.type(torch.LongTensor).to(self.device))

    def get_center_losses(self, center_label, predicted_center, stage1_center):
        #center_distance = Variable(torch.norm(center_label - predicted_center, p=1, dim=-1), requires_grad= True)
        center_distance = torch.norm(center_label - predicted_center, p=1, dim=-1)
        center_loss = self.huber_loss(center_distance, delta=2.0)
        #stage1_center_distance = Variable(torch.norm(center_label - stage1_center, p=1, dim=-1), requires_grad= True)
        stage1_center_distance = torch.norm(center_label - stage1_center, p=1, dim=-1)

        stage1_center_loss = self.huber_loss(stage1_center_distance, delta=1.0)
        return center_loss, stage1_center_loss

    def get_heading_loss(self, heading_class_label, heading_scores, heading_residual_label,
                         heading_residuals_normalized):
        heading_class_loss = F.cross_entropy(heading_scores,
                                             heading_class_label.type(torch.LongTensor).to(self.device))

        # to fix later one hot encoder
        one_hot_heading_class_label = convert_to_one_hot(heading_class_label, self.num_heading_bin, self.device)

        heading_residuals_normalized_label = heading_residual_label / (np.pi / self.num_heading_bin)

        heading_residuals_normalized_loss = self.huber_loss(torch.sum(
            heading_residuals_normalized * one_hot_heading_class_label, dim=1) - heading_residuals_normalized_label, delta=1.0)

        return heading_class_loss, heading_residuals_normalized_loss, one_hot_heading_class_label

    def get_size_loss(self, size_scores, size_class_label, size_residual_label, size_residual_normalized):
        size_class_loss = F.cross_entropy(size_scores,
                                          size_class_label.type(torch.LongTensor).to(self.device))

        # to be fixed later
        one_hot_size_class_labels = convert_to_one_hot(size_class_label, self.num_size_cluster)

        #
        one_hot_labels_rep = one_hot_size_class_labels.view(-1, self.num_size_cluster, 1).repeat(1, 1, 3)

        #
        predicted_size_residual_normalized = torch.sum(size_residual_normalized * one_hot_labels_rep, dim=1)

        mean_size_arr = self.mean_size_array.view(-1, self.num_size_cluster, 3)

        mean_size_label = torch.sum(one_hot_labels_rep * mean_size_arr, dim=1)  # Bx3
        size_residual_label_normalized = size_residual_label / mean_size_label

        size_normalized_distance = torch.norm(size_residual_label_normalized - predicted_size_residual_normalized,
                                              p=1, dim=-1)

        size_residual_normalized_loss = self.huber_loss(size_normalized_distance, delta=1.0)

        return size_class_loss, size_residual_normalized_loss, one_hot_size_class_labels

    def get_corner_loss(self, center_label, center, heading_residual_label, heading_residuals, size_residuals_label,
                        size_residuals):
        # B X NH X NS X 8 X 3 (each box is 8 X 3)
        corners_3d = get_box3d_corners(center, heading_residuals, size_residuals,
                                       self.num_heading_bin, self.num_size_cluster, self.mean_size_array, self.device)

        # final shape B X NH X NS
        # one hot size class    --> B X NS
        # one hot heading class --> B X NH
        gt_mask = self.one_hot_hc_label \
                      .unsqueeze(dim=2) \
                      .repeat(1, 1, self.num_size_cluster) * \
                  self.one_hot_sc_label. \
                      unsqueeze(dim=1) \
                      .repeat(1, self.num_heading_bin, 1)

        # final shape B X 8 X 3
        # corners_3d    --> B X NH X NS X 8 X 3
        # gt_mask       --> B X NH X NS
        corners_3d_predicted = torch.sum(gt_mask.unsqueeze(3).unsqueeze(4) * corners_3d, dim=[1, 2])

        # shape is (NH,)
        heading_bin_centers = torch.Tensor(
            np.arange(0, 2 * np.pi, 2 * np.pi / self.num_heading_bin)).type(torch.FloatTensor).to(self.device)

        # B X NH = B X 1 + 1 X NH (broadcasting both dimensions)
        heading_label = heading_residual_label.unsqueeze(1) + heading_bin_centers.unsqueeze(0)

        heading_label = torch.sum(self.one_hot_hc_label * heading_label, dim=1)

        mean_sizes = self.mean_size_array.unsqueeze(0)

        # 1 X NS X 3 + B X 1 X 3 = B X NS X 3
        size_label = mean_sizes + size_residuals_label.unsqueeze(1)

        # B X 3
        # one hot       --> B X NS X 1
        # size_label    --> B X NS X 3
        size_label = torch.sum(self.one_hot_sc_label.unsqueeze(2) * size_label, dim=1)

        # B X 8 X 3
        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label+np.pi, size_label)

        corners_dist = torch.min(torch.norm(corners_3d_predicted - corners_3d_gt, p=1, dim=-1),
                                 torch.norm(corners_3d_predicted - corners_3d_gt_flip, p=1, dim=-1))

        corner_loss = self.huber_loss(corners_dist, delta=1.0)

        return corner_loss







