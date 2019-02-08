import numpy as np
from Frustum3DModel import Frustum3DModel
from provider import FrustumDataset, compute_box3d_iou
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from model_utils import save_checkpoint, load_checkpoint
from train_utils import get_batch
import logging
from datetime import datetime
import torch

class ModelTrainer:
    def __init__(self, model: Frustum3DModel,
                 train_dataset: FrustumDataset,
                 valid_dataset: FrustumDataset,
                 config,
                 device='cuda',
                 train_subset=None,
                 log_interval=10):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.config = config
        self.train_batch_size = config.BATCH_SIZE
        self.val_batch_size = config.BATCH_SIZE

        self.device = device
        self.loss_func = config.loss_function
        self.loss = self.loss_func(self.config.NUM_HEADING_BIN, self.config.NUM_SIZE_CLUSTER, self.model.endpoints,
                                   self.config)

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
        self.n_epochs = 0
        self.global_step = 0
        self.train_control = config.train_control
        self.current_lr = config.train_control['optimizer_params']['lr']
        self.bn_decay = config.train_control['init_bn_decay']

        self._reset_histories()
        self.best_model = None
        self.best_val_loss = 100000
        self.endpoints = self.model.endpoints

        self.lr_decay_cycle = 1
        self.bn_decay_cycle = 1
        self.logger_name = self.__class__.__name__
        if not hasattr(self, 'columns'):
            self.columns = '''
                    epoch | batches_processed | mean_loss | segmentation_accuracy | box_IOU_ground | box_IOU_3d | box_accuracy | seg_loss | stage1_center_loss | center_loss | heading_class_loss | heading_residual_normalized_loss | size_class_loss | size_residuals_normalized_loss | corner_loss | total_loss | lr | bn_decay | flag
                    '''
        self._init_logger()


    def _init_logger(self):

        self.df_logger = logging.getLogger(self.logger_name)
        self.df_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            datetime.now().strftime(
                f"./logs/%Y-%m-%d_%H-%M-%S-{self.__class__.__name__}.log"), 'w+')

        self.df_logger.addHandler(handler)
        self.df_logger.addHandler(logging.StreamHandler())

        self.df_logger.info('Logging initialized')
        self.df_logger.info(self.columns)

    def _init_optimizer(self, train_control):

        self.optimizer = train_control['optimizer'](self.get_trainable_parameters(),
                                                    **train_control['optimizer_params'])

        # if train_control['lr_scheduler_type'] == 'step':
        #     self.scheduler = StepLR(self.optimizer, **train_control['step_scheduler_args'])
        # elif train_control['lr_scheduler_type'] == 'exp':
        #     self.scheduler = ExponentialLR(self.optimizer, **train_control['exp_scheduler_args'])
        # elif train_control['lr_scheduler_type'] == 'plateau':
        #     self.scheduler = ReduceLROnPlateau(self.optimizer, **train_control['plateau_scheduler_args'])
        # else:
        #     self.scheduler = StepLR(self.optimizer, step_size=100, gamma=1)

    def _reset_histories(self):
        return True

    def get_trainable_parameters(self):
        model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        loss_params = self.loss.get_trainable_weights() if hasattr(self.loss, 'get_trainable_weights') else []
        return model_params + loss_params


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
            # self.loss.losses['seg_loss'].backward()
            # self.loss.losses['size_class_loss'].backward()
            # self.loss.losses['heading_residual_normalized_loss'].backward()
            # self.loss.losses['size_residuals_normalized_loss'].backward()
            # self.loss.losses['stage1_center_loss'].backward()
            # self.loss.losses['corner_loss'].backward()
            # self.loss.losses['center_loss'].backward()

            self.optimizer.step()

            # print("after backward: ", type(self.endpoints))
            # print("after backward: ", self.endpoints.keys())

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

            if (batch_idx + 1) % self.log_interval == 0:
                seg_acc = (total_correct / float(total_seen))
                iou_ground = iou2ds_sum / float(self.train_batch_size * self.log_interval)
                iou_3d = iou3ds_sum / float(self.train_batch_size * self.log_interval)

                box_acc = float(iou3d_correct_cnt) / float(self.train_batch_size * self.log_interval)

                self.log_values(batch_idx, loss_sum / self.log_interval, seg_acc, iou_ground, iou_3d, box_acc, 'Train')

                total_correct = 0
                total_seen = 0
                loss_sum = 0
                iou2ds_sum = 0
                iou3ds_sum = 0
                iou3d_correct_cnt = 0


    def eval_epoch(self):
        self.model.eval()
        test_idxs = np.arange(0, len(self.valid_dataset))
        num_batches = len(self.valid_dataset) // self.val_batch_size

        # To collect statistics
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(self.config.NUM_CLASSES)]
        total_correct_class = [0 for _ in range(self.config.NUM_CLASSES)]
        iou2ds_sum = 0
        iou3ds_sum = 0
        iou3d_correct_cnt = 0

        # Simple evaluation with batches
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
                self.endpoints = self.model(batch_data, batch_one_hot_vec)
                val_loss = self.loss(batch_label,
                                       batch_center,
                                       batch_hclass, batch_hres,
                                       batch_sclass, batch_sres)

            preds_val = np.argmax(self.endpoints['mask_logits'].detach().cpu().numpy(), 2)
            correct = np.sum(preds_val == batch_label.detach().cpu().numpy())
            total_correct += correct
            total_seen += (self.val_batch_size * self.config.NUM_POINT)
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

            for l in range(self.config.NUM_CLASSES):
               total_seen_class[l] += np.sum(batch_label.detach().cpu().numpy() == l)
               total_correct_class[l] += (np.sum((preds_val == l) & (batch_label.detach().cpu().numpy() == l)))
        seg_acc = (total_correct / float(total_seen))
        iou_ground = iou2ds_sum / float(self.val_batch_size * num_batches)
        iou_3d = iou3ds_sum / float(self.val_batch_size * num_batches)

        box_acc = float(iou3d_correct_cnt) / float(self.val_batch_size * num_batches)

        self.log_values(batch_idx, loss_sum / float(num_batches), seg_acc, iou_ground, iou_3d, box_acc, 'Val')

        if self.best_val_loss > (loss_sum / float(num_batches)):
            self.best_val_loss = (loss_sum / float(num_batches))
            self.best_model = self.model
            save_checkpoint('./models/best_model.pth', self.model, self.epoch, self.optimizer, self.best_val_loss)


    def log_values(self, batch_idx, mean_loss, seg_acc, iou_ground, iou_3d, box_acc, flag = 'Train'):
        '''
        timestamp | epoch | batches_processed | mean_loss | segmentation_accuracy | box_IOU_ground | box_IOU_3d |
        box_accuracy | seg_loss | stage1_center_loss | center_loss | heading_class_loss |
        heading_residual_normalized_loss | size_class_loss | size_residuals_normalized_loss | corner_loss | total_loss
        '''

        log_int = lambda x, y=True: '%d' % x + ' | ' if y else '%f' % x
        log_float = lambda x, y=True: '%f' % x + ' | ' if y else '%f' % x
        log_str = lambda  x, y=True: x + ' | ' if y else x

        log_string = ''
        log_string += log_int(self.epoch)
        log_string += log_int((batch_idx + 1))
        log_string += log_float(mean_loss)
        log_string += log_float(seg_acc)
        log_string += log_float(iou_ground)
        log_string += log_float(iou_3d)
        log_string += log_float(box_acc)
        log_string += log_float(self.loss.losses['seg_loss'])
        log_string += log_float(self.loss.losses['stage1_center_loss'])
        log_string += log_float(self.loss.losses['center_loss'])
        log_string += log_float(self.loss.losses['heading_class_loss'])
        log_string += log_float(self.loss.losses['heading_residual_normalized_loss'])
        log_string += log_float(self.loss.losses['size_class_loss'])
        log_string += log_float(self.loss.losses['size_residuals_normalized_loss'])
        log_string += log_float(self.loss.losses['corner_loss'])
        log_string += log_float(self.loss.losses['total_loss'])
        log_string += log_float(self.current_lr)
        log_string += log_float(self.bn_decay)
        log_string += log_str(flag, False)

        self.df_logger.info(log_string + '\n')

    def train(self, n_epochs):
        self.model.to(self.device)
        self.model.train()
        self.n_epochs = n_epochs

        for epoch in range(n_epochs):
            self.train_epoch()
            self.eval_epoch()
            self.exp_lr_scheduler()
            self.exp_bn_scheduler()
            if self.config.variable_loss_weights:
                self.update_losses_weights()
            self.epoch += 1




    def update_losses_weights(self):
        self.loss.change_loss_components_weights(self.n_epochs)

    def resume_training(self, n_epochs, model_path='./models/best_model.pth'):
        self.model, self.optimizer, self.epoch, self.best_val_loss = load_checkpoint(
            model_path,
            self.model,
            self.optimizer)

        # to verify and store optimizer parameters instead of re-init
        self._init_optimizer(self.train_control)
        print("Loaded model and resuming training...")

        self.train(n_epochs)


    def exp_lr_scheduler(self, staircase=True):
        """Decay learning rate by a factor """

        if staircase:
            lr = self.train_control['optimizer_params']['lr'] * self.train_control['decay_rate'] ** (
                (self.global_step * self.config.BATCH_SIZE) // self.train_control['decay_steps'])
        else:
            lr = self.train_control['optimizer_params']['lr'] * self.train_control['decay_rate'] ** (
                    (self.global_step * self.config.BATCH_SIZE) / self.train_control['decay_steps'])
        lr = max(lr, self.train_control['lr_clip'])

        decay_condition = (self.global_step * self.config.BATCH_SIZE) //\
                          self.train_control['decay_steps'] == self.lr_decay_cycle
        if decay_condition:
            print('LR is set to {}'.format(lr))
            self.current_lr = lr
            self.lr_decay_cycle += 1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        print("Current Learning Rate is: ", self.current_lr)


    def exp_bn_scheduler(self, staircase=True):
        """Decay batch norm by a factor """
        if staircase:
            bn_decay = self.train_control['init_bn_decay'] * self.train_control['bn_decay_rate'] ** (
                (self.global_step * self.config.BATCH_SIZE) // self.train_control['bn_decay_step'])
        else:
            bn_decay = self.train_control['init_bn_decay'] * self.train_control['bn_decay_rate'] ** (
                    (self.global_step * self.config.BATCH_SIZE) / self.train_control['bn_decay_step'])

        bn_decay = min(1 - bn_decay, self.train_control['bn_decay_clip'])

        decay_condition = (self.global_step * self.config.BATCH_SIZE) //\
                          self.train_control['bn_decay_step'] == self.bn_decay_cycle

        if decay_condition:
            print('Batch norm decay is set to {}'.format(bn_decay))
            self.bn_decay = bn_decay
            self.bn_decay_cycle += 1
            self.model.update_bn_decay(self.bn_decay)

        print("Current Batch norm decay is: ", self.bn_decay)


