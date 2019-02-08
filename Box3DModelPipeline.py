import logging
import torch
import torch.nn as nn
import numpy as np
from Trainer import ModelTrainer
from train_utils import get_batch
from CenterRegressionModel import CenterRegressionModel
import torch.nn.functional as F
from provider import compute_box3d_iou
from RegressionBox3DModel import RegressionBox3DModel
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from model_utils import point_cloud_masking, parse_3dregression_model_output, convert_to_one_hot
from model_utils import get_mean_size_array, get_box3d_corners, get_box3d_corners_helper



class Box3DModelBlock(nn.Module):
    def __init__(self, n_points, m_points, n_channels, n_classes, device='cuda', batch_norm_decay=0.9,
                 num_heading_bin=12, num_size_cluster=8, resample_method='random'):
        super(Box3DModelBlock, self).__init__()

        self.resample_method = resample_method
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.endpoints = {}
        self.m_points = m_points
        self.device = device

        self.center_regression_model = CenterRegressionModel(m_points, n_channels=n_channels, n_classes=n_classes,
                                                             batch_norm_decay=batch_norm_decay)

        self.regression_box3d_model = RegressionBox3DModel(m_points, n_channels=n_channels, n_classes=n_classes,
                                                           batch_norm_decay=batch_norm_decay,
                                                           num_heading_bin=num_heading_bin,
                                                           num_size_cluster=num_size_cluster)

    def forward(self, input_point_cloud, one_hot_vector, segmentation_label):
        segmentation_label = convert_to_one_hot(segmentation_label, 2)
        self.object_point_cloud, self.mask_mean_xyz, self.endpoints = point_cloud_masking(input_point_cloud,
                                                                                          segmentation_label,
                                                                                          self.endpoints,
                                                                                          self.m_points,
                                                                                          resample_method=self.resample_method)

        self.predicted_center_delta = self.center_regression_model(self.object_point_cloud, one_hot_vector)

        self.endpoints['stage1_center'] = self.predicted_center_delta + self.mask_mean_xyz

        self.object_point_cloud[:, :, 0:3] = self.object_point_cloud[:, :, 0:3] - self.predicted_center_delta.unsqueeze(
            1)

        self.boxmodel_output = self.regression_box3d_model(self.object_point_cloud, one_hot_vector)

        self.endpoints = parse_3dregression_model_output(self.boxmodel_output, self.endpoints, self.num_heading_bin,
                                                         self.num_size_cluster, self.device)

        self.endpoints['center'] = self.endpoints['center_boxnet'] + self.endpoints['stage1_center']

        return self.endpoints

    def update_bn_decay(self, current_bn_decay):
        for module in self.modules():
            if isinstance(module, BatchNorm2d) or isinstance(module, BatchNorm1d):
                module.momentum = 1 - current_bn_decay

# -------------------------------------------------------------------------------------------------------------

class Box3DModelLoss(nn.Module):
    def __init__(self, num_heading_bin, num_size_cluster, endpoints, config, device='cuda'):
        super(Box3DModelLoss, self).__init__()
        self.device = device

        self.seg_loss_weight = config.seg_loss_weight
        self.corner_loss_weight = config.corner_loss_weight
        self.box_loss_weight = config.box_loss_weight

        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.one_hot_hc_label = None
        self.one_hot_sc_label = None
        self.mean_size_array = torch.Tensor(get_mean_size_array(self.num_size_cluster)).to(self.device)
        self.losses = {}
        self.endpoints = endpoints



    def forward(self, center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label):

        endpoints = self.endpoints

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

        total_loss = self.box_loss_weight * (
                center_loss + heading_class_loss + size_class_loss +
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


# -------------------------------------------------------------------------------------------------------------

class Box3DModelTrainer(ModelTrainer):
    def __init__(self, model: Box3DModelBlock,
                 train_dataset,
                 valid_dataset,
                 config,
                 device='cuda',
                 train_subset=None,
                 log_interval=10):
        self.columns = '''epoch | batches_processed | mean_loss | box_accuracy | corner_loss | center_loss | 
stage1_center_loss | heading_class_loss | heading_residual_normalized_loss | size_class_loss | 
size_residuals_normalized_loss | total_loss | lr | bn_decay | flag'''

        super(Box3DModelTrainer, self).__init__(model, train_dataset, valid_dataset,
                                                config, device, train_subset, log_interval)



    def log_box_values(self, batch_idx, mean_loss, box_acc, flag='Train'):
        log_int = lambda x, y=True: '%d' % x + ' | ' if y else '%f' % x
        log_float = lambda x, y=True: '%f' % x + ' | ' if y else '%f' % x
        log_str = lambda x, y=True: x + ' | ' if y else x

        log_string = ' '
        log_string += log_int(self.epoch)
        log_string += log_int((batch_idx + 1))
        log_string += log_float(mean_loss)
        log_string += log_float(box_acc)
        log_string += log_float(self.loss.losses['corner_loss'])
        log_string += log_float(self.loss.losses['center_loss'])
        log_string += log_float(self.loss.losses['stage1_center_loss'])
        log_string += log_float(self.loss.losses['heading_class_loss'])
        log_string += log_float(self.loss.losses['heading_residual_normalized_loss'])
        log_string += log_float(self.loss.losses['size_class_loss'])
        log_string += log_float(self.loss.losses['size_residuals_normalized_loss'])
        log_string += log_float(self.loss.losses['total_loss'])
        log_string += log_float(self.current_lr)
        log_string += log_float(self.bn_decay)
        log_string += log_str(flag, False)

        self.df_logger.info(log_string + '\n')

    def train_epoch(self):

        train_idxs = np.arange(0, self.train_dataset_length)
        np.random.shuffle(train_idxs)

        loss_sum = 0
        iou2ds_sum = 0
        iou3ds_sum = 0

        iou3d_correct_cnt = 0

        for batch_idx in range(self.num_batches):
            self.global_step += 1
            start_idx = batch_idx * self.train_batch_size
            end_idx = (batch_idx + 1) * self.train_batch_size

            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                tuple(get_batch(self.train_dataset, train_idxs, start_idx, end_idx,
                                self.config.NUM_POINT, self.config.NUM_CHANNELS))
            self.model.zero_grad()
            self.endpoints = self.model(batch_data, batch_one_hot_vec, batch_label)
            total_loss = self.loss(batch_center,
                                   batch_hclass, batch_hres,
                                   batch_sclass, batch_sres)

            total_loss.backward()
            self.optimizer.step()

            loss_sum += total_loss

            iou2ds, iou3ds = compute_box3d_iou(self.endpoints['center'].detach().cpu().numpy(),
                                               self.endpoints['heading_scores'].detach().cpu().numpy(),
                                               self.endpoints['heading_residuals'].detach().cpu().numpy(),
                                               self.endpoints['size_scores'].detach().cpu().numpy(),
                                               self.endpoints['size_residuals'].detach().cpu().numpy(),
                                               batch_center.detach().cpu().numpy(),
                                               batch_hclass.detach().cpu().numpy(),
                                               batch_hres.detach().cpu().numpy(),
                                               batch_sclass.detach().cpu().numpy(),
                                               batch_sres.detach().cpu().numpy())
            self.endpoints['iou2ds'] = iou2ds
            self.endpoints['iou3ds'] = iou3ds

            iou2ds_sum += np.sum(self.endpoints['iou2ds'])
            iou3ds_sum += np.sum(self.endpoints['iou3ds'])
            iou3d_correct_cnt += np.sum(self.endpoints['iou3ds'] >= 0.7)

            if (batch_idx + 1) % self.log_interval == 0:

                box_acc = float(iou3d_correct_cnt) / float(self.train_batch_size * self.log_interval)
                self.log_box_values(batch_idx, loss_sum / self.log_interval, box_acc, 'Train')

                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0

    def eval_epoch(self):
        self.model.eval()
        test_idxs = np.arange(0, len(self.valid_dataset))
        num_batches = len(self.valid_dataset) // self.val_batch_size

        # To collect statistics
        loss_sum = 0
        iou2ds_sum = 0
        iou3ds_sum = 0
        iou3d_correct_cnt = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.val_batch_size
            end_idx = (batch_idx + 1) * self.val_batch_size

            batch_data, batch_label, batch_center, \
            batch_hclass, batch_hres, \
            batch_sclass, batch_sres, \
            batch_rot_angle, batch_one_hot_vec = \
                tuple(get_batch(self.valid_dataset, test_idxs, start_idx, end_idx,
                                self.config.NUM_POINT, self.config.NUM_CHANNELS))

            with torch.no_grad():
                self.endpoints = self.model(batch_data, batch_one_hot_vec, batch_label)
                val_loss = self.loss(batch_center,
                                       batch_hclass, batch_hres,
                                       batch_sclass, batch_sres)

            loss_sum += val_loss

            iou2ds, iou3ds = compute_box3d_iou(self.endpoints['center'].detach().cpu().numpy(),
                                               self.endpoints['heading_scores'].detach().cpu().numpy(),
                                               self.endpoints['heading_residuals'].detach().cpu().numpy(),
                                               self.endpoints['size_scores'].detach().cpu().numpy(),
                                               self.endpoints['size_residuals'].detach().cpu().numpy(),
                                               batch_center.detach().cpu().numpy(),
                                               batch_hclass.detach().cpu().numpy(),
                                               batch_hres.detach().cpu().numpy(),
                                               batch_sclass.detach().cpu().numpy(),
                                               batch_sres.detach().cpu().numpy())
            self.endpoints['iou2ds'] = iou2ds
            self.endpoints['iou3ds'] = iou3ds

            iou2ds_sum += np.sum(self.endpoints['iou2ds'])
            iou3ds_sum += np.sum(self.endpoints['iou3ds'])
            iou3d_correct_cnt += np.sum(self.endpoints['iou3ds'] >= 0.7)

        box_acc = float(iou3d_correct_cnt) / float(self.val_batch_size * num_batches)
        self.log_box_values(batch_idx, loss_sum / float(num_batches), box_acc, 'Val')

        if self.best_val_loss > (loss_sum / float(num_batches)):
            self.best_val_loss = (loss_sum / float(num_batches))
            self.best_model = self.model

    def train(self, n_epochs):
        self.model.to(self.device)
        self.model.train()
        self.n_epochs = n_epochs

        for epoch in range(n_epochs):
            self.train_epoch()
            self.eval_epoch()
            self.exp_lr_scheduler()
            self.exp_bn_scheduler()
            self.epoch += 1