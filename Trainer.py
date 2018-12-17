import numpy as np
from Frustum3DModel import Frustum3DModel
from provider import FrustumDataset, compute_box3d_iou
from Frustum3DLoss import Frustum3DLoss
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from train_utils import get_batch
import logging
from datetime import datetime
import pprint



class ModelTrainer:
    def __init__(self, model: Frustum3DModel,
                 train_dataset: FrustumDataset,
                 valid_dataset: FrustumDataset,
                 config,
                 device='cuda',
                 loss_func=Frustum3DLoss,
                 train_subset=None,
                 log_interval=10):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.config = config
        self.train_batch_size = config.BATCH_SIZE

        self.device = device
        self.loss_func = loss_func
        self.loss = self.loss_func(self.config.NUM_HEADING_BIN, self.config.NUM_SIZE_CLUSTER, self.model.endpoints)
        self.log_interval = log_interval
        self.train_subset = train_subset

        if self.train_subset:
            self.train_dataset_length = self.train_subset
        else:
            self.train_dataset_length = len(train_dataset)

        self.num_batches = self.train_dataset_length // self.train_batch_size

        self.optimizer = None
        self.scheduler = None
        self._init_optimizer(config.train_control)
        self.epoch = 0
        self.global_step = 0
        self.train_control = config.train_control
        self.bn_decay = config.train_control['init_bn_decay']

        self._reset_histories()
        self.best_model = None
        self.endpoints = self.model.endpoints

        handlers = [logging.FileHandler(
            datetime.now().strftime(
                f"./logs/%Y-%m-%d_%H-%M-%S-.log")),
            logging.StreamHandler()]
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            level=logging.INFO, handlers=handlers)

        logging.info('Logging initialized')

        logging.info('''
        timestamp | epoch | batches_processed | mean loss | segmentation accuracy | box IOU ground | box IOU 3d | box accuracy | seg_loss
        ''')

    def _init_optimizer(self, train_control):

        self.optimizer = train_control['optimizer'](filter(lambda p: p.requires_grad, self.model.parameters()),
                                                    **train_control['optimizer_params'])

        if train_control['lr_scheduler_type'] == 'step':
            self.scheduler = StepLR(self.optimizer, **train_control['step_scheduler_args'])
        elif train_control['lr_scheduler_type'] == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, **train_control['exp_scheduler_args'])
        elif train_control['lr_scheduler_type'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, **train_control['plateau_scheduler_args'])
        else:
            self.scheduler = StepLR(self.optimizer, step_size=100, gamma=1)

    def _reset_histories(self):
        return True

    def train_epoch(self):
        # this is the current iteration inside the epoch

        train_idxs = np.arange(0, self.train_dataset_length)
        np.random.shuffle(train_idxs)

        # To collect statistics
        total_correct = 0
        total_seen = 0
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
            self.endpoints = self.model(batch_data, batch_one_hot_vec)
            total_loss = self.loss(batch_label,
                                   batch_center,
                                   batch_hclass, batch_hres,
                                   batch_sclass, batch_sres)

            total_loss.backward()
            #self.loss.losses['seg_loss'].backward()
            #self.loss.losses['size_class_loss'].backward()
            #self.loss.losses['heading_residual_normalized_loss'].backward()
            #self.loss.losses['size_residuals_normalized_loss'].backward()
            #self.loss.losses['stage1_center_loss'].backward()
            #self.loss.losses['corner_loss'].backward()
            #self.loss.losses['center_loss'].backward()
            
            self.optimizer.step()
            self.exp_lr_scheduler()
            self.exp_bn_scheduler()

            #print("after backward: ", type(self.endpoints))
            #print("after backward: ", self.endpoints.keys())

            preds_val = np.argmax(self.endpoints['mask_logits'].detach().cpu().numpy(), 2)
            correct = np.sum(preds_val == batch_label.detach().cpu().numpy())
            total_correct += correct
            total_seen += (self.train_batch_size * self.config.NUM_POINT)
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

            self.global_step += 1

            if (batch_idx + 1) % self.log_interval == 0:
                seg_acc = (total_correct / float(total_seen))
                iou_ground  = iou2ds_sum / float(self.train_batch_size * self.log_interval)
                iou_3d = iou3ds_sum / float(self.train_batch_size * self.log_interval)

                box_acc = float(iou3d_correct_cnt) / float(self.train_batch_size * self.log_interval)

                self.log_values(batch_idx, loss_sum/self.log_interval, seg_acc, iou_ground, iou_3d, box_acc)

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0

        self.epoch += 1

    def log_values(self, batch_idx, mean_loss, seg_acc, iou_ground, iou_3d, box_acc):
        '''
        epoch | batches_processed | mean loss | segmentation accuracy | box IOU(ground | box IOU 3d | box accuracy | losses
        '''

        log_string = '|'
        log_string += '%d' % self.epoch + '|'
        log_string += '%d' % (batch_idx + 1) + '|'
        log_string += '%f' % mean_loss + '|'
        log_string += '%f' % seg_acc + '|'
        log_string += '%f' % iou_ground + '|'
        log_string += '%f' % iou_3d + '|'
        log_string += '%f' % box_acc + '|'
        log_string += '%f' % self.loss.losses['seg_loss']

        logging.info(log_string+'\n')

    def train(self, n_epochs):
        self.model.to(self.device)
        self.model.train()

        for epoch in range(n_epochs):
            self.train_epoch()

    def exp_lr_scheduler(self, staircase=True):
        """Decay learning rate by a factor """
        if staircase:
            lr = self.train_control['optimizer_params']['lr'] * self.train_control['decay_rate'] ** (
                    self.global_step // self.train_control['decay_steps'])
        else:
            lr = self.train_control['optimizer_params']['lr'] * self.train_control['decay_rate'] ** (
                    self.global_step / self.train_control['decay_steps'])
        lr = max(lr, self.train_control['lr_clip'])

        if self.global_step % self.train_control['decay_steps'] == 0:
            print('LR is set to {}'.format(lr))
            logging.info('LR is set to {}'.format(lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def exp_bn_scheduler(self, staircase=True):
        """Decay batch norm by a factor """
        if staircase:
            bn_decay = self.train_control['init_bn_decay'] * self.train_control['bn_decay_rate'] ** (
                    self.global_step // self.train_control['bn_decay_step'])
        else:
            bn_decay = self.train_control['init_bn_decay'] * self.train_control['bn_decay_rate'] ** (
                    self.global_step / self.train_control['bn_decay_step'])

        bn_decay = min(1 - bn_decay, self.train_control['bn_decay_clip'])

        if self.global_step % self.train_control['bn_decay_step'] == 0:
            print('Batch norm decay is set to {}'.format(bn_decay))
            logging.info('Batch norm decay is set to {}'.format(bn_decay))

        self.bn_decay = bn_decay
        self.model.update_bn_decay(self.bn_decay)
