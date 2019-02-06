import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Trainer import ModelTrainer
from train_utils import get_batch
import logging


from Segmentation3DModel import Segmentation3DModel

class Segmentation3DModelBlock(Segmentation3DModel):
    def __init__(self, n_points, n_channels, n_classes, batch_norm_decay=0.9):
        super(Segmentation3DModelBlock, self).__init__(n_points, n_channels, n_classes, batch_norm_decay)
        self.endpoints = {}

    def forward(self, input_point_cloud: torch.Tensor, one_hot_vector: torch.Tensor):
        self.logits = super().forward(input_point_cloud, one_hot_vector)
        return self.logits

# -------------------------------------------------------------------------------------------------------------

class Segmentation3DModelLoss(nn.Module):
    def __init__(self, num_heading_bin, num_size_cluster, endpoints, config, device='cuda'):
        super(Segmentation3DModelLoss, self).__init__()
        self.device = device

    def forward(self, logits, mask_label):
        seg_loss = self.get_segmentation_loss(mask_label, logits)
        return seg_loss

    def get_segmentation_loss(self, mask_label, segmentation_logits):
        return F.cross_entropy(segmentation_logits.permute(0, 2, 1),
                               mask_label.type(torch.LongTensor).to(self.device))

# -------------------------------------------------------------------------------------------------------------

class Segmentation3DModelTrainer(ModelTrainer):
    def __init__(self, model: Segmentation3DModelBlock,
                 train_dataset,
                 valid_dataset,
                 config,
                 device='cuda',
                 train_subset=None,
                 log_interval=10):
        self.columns = '''epoch | batches_processed | seg_acc | seg_loss | lr | bn_decay | flag'''
        super(Segmentation3DModelTrainer, self).__init__(model, train_dataset, valid_dataset,
                                                         config, device, train_subset, log_interval)

    def log_seg_values(self, batch_idx, seg_loss, seg_acc, flag = 'Train'):

        log_int = lambda x, y=True: '%d' % x + ' | ' if y else '%f' % x
        log_float = lambda x, y=True: '%f' % x + ' | ' if y else '%f' % x
        log_str = lambda  x, y=True: x + ' | ' if y else x

        log_string = ' '
        log_string += log_int(self.epoch)
        log_string += log_int((batch_idx + 1))
        log_string += log_float(seg_acc)
        log_string += log_float(seg_loss)
        log_string += log_float(self.current_lr)
        log_string += log_float(self.bn_decay)
        log_string += log_str(flag, False)

        self.df_logger.info(log_string + '\n')

    def train_epoch(self):

        train_idxs = np.arange(0, self.train_dataset_length)
        np.random.shuffle(train_idxs)

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(self.num_batches):
            self.global_step += 1
            start_idx = batch_idx * self.train_batch_size
            end_idx = (batch_idx + 1) * self.train_batch_size

            batch_data, batch_label, _, \
                    _, _, \
            _, _, \
            _, batch_one_hot_vec = \
                tuple(get_batch(self.train_dataset, train_idxs, start_idx, end_idx,
                                self.config.NUM_POINT, self.config.NUM_CHANNELS))
            self.model.zero_grad()
            self.logits = self.model(batch_data, batch_one_hot_vec)
            total_loss = self.loss(self.logits, batch_label)

            total_loss.backward()

            self.optimizer.step()

            preds_val = np.argmax(self.logits.detach().cpu().numpy(), 2)
            correct = np.sum(preds_val == batch_label.detach().cpu().numpy())
            total_correct += correct
            total_seen += (self.train_batch_size * self.config.NUM_POINT)
            loss_sum += total_loss

            if (batch_idx + 1) % self.log_interval == 0:
                seg_acc = (total_correct / float(total_seen))

                self.log_seg_values(batch_idx, loss_sum / self.log_interval, seg_acc, 'Train')

                total_correct = 0
                total_seen = 0
                loss_sum = 0

    def eval_epoch(self):
        self.model.eval()
        test_idxs = np.arange(0, len(self.valid_dataset))
        num_batches = len(self.valid_dataset) // self.val_batch_size

        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.config.NUM_CLASSES)]
        total_correct_class = [0 for _ in range(self.config.NUM_CLASSES)]

        # Simple evaluation with batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.val_batch_size
            end_idx = (batch_idx + 1) * self.val_batch_size

            batch_data, batch_label, _, \
            _, _, \
            _, _, \
            _, batch_one_hot_vec = \
                tuple(get_batch(self.valid_dataset, test_idxs, start_idx, end_idx,
                          self.config.NUM_POINT, self.config.NUM_CHANNELS))

            with torch.no_grad():
                self.logits = self.model(batch_data, batch_one_hot_vec)
                val_loss = self.loss(self.logits, batch_label)

            preds_val = np.argmax(self.logits.detach().cpu().numpy(), 2)
            correct = np.sum(preds_val == batch_label.detach().cpu().numpy())
            total_correct += correct
            total_seen += (self.val_batch_size * self.config.NUM_POINT)
            loss_sum += val_loss

            for l in range(self.config.NUM_CLASSES):
               total_seen_class[l] += np.sum(batch_label.detach().cpu().numpy() == l)
               total_correct_class[l] += (np.sum((preds_val == l) & (batch_label.detach().cpu().numpy() == l)))
        seg_acc = (total_correct / float(total_seen))

        self.log_seg_values(batch_idx, loss_sum / float(num_batches), seg_acc, 'Val')

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
