from torch.optim import SGD, Adam
from Frustum3DLoss import Frustum3DLoss
from Frustum3DTrainableLoss import Frustum3DTrainableLoss

# Config
class Config:

    NUM_HEADING_BIN = 12
    NUM_SIZE_CLUSTER = 8
    NUM_POINT = 1024
    NUM_CHANNELS = 4
    NUM_OBJECT_POINT = 512
    MOMENTUM = 0.9
    DECAY_STEP = 800000
    DECAY_RATE = 0.5

    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    loss_function = Frustum3DTrainableLoss
    variable_loss_weights = False

    # all not active if variable loss weights is False
    start_seg_loss_weight = 1.0
    start_corner_loss_weight = 20.0
    start_box_loss_weight = 1.0

    # end values in case variable loss is switched on, else default values during the whole training
    seg_loss_weight = 1.0
    corner_loss_weight = 10.0
    box_loss_weight = 1.0

    BATCH_SIZE = 64
    NUM_CLASSES = 2  # segmentation has 2 classes

    resampling_method = 'random' # 'repeat'

    train_control = {
        'optimizer': Adam,  # Adam, SGD
        'optimizer_params': {'lr': 1e-3},
        'decay_steps': DECAY_STEP,
        'decay_rate': DECAY_RATE,
        'lr_clip': 5e-6,
        'init_bn_decay': BN_INIT_DECAY,
        'bn_decay_rate': BN_DECAY_RATE,
        'bn_decay_step': BN_DECAY_STEP,
        'bn_decay_clip': BN_DECAY_CLIP,
        'lr_scheduler_type': 'exp',

        'step_scheduler_args': {
            'gamma': 0.7,  # factor to decay learing rate (new_lr = gamma * lr)
            'step_size': 3  # number of epochs to take a step of decay
        },

        'exp_scheduler_args': {
            'gamma': 0.1  # factor to decay learing rate (new_lr = gamma * lr)
        },

        'plateau_scheduler_args': {
            'factor': 0.2,  # factor to decay learing rate (new_lr = factor * lr)
            'patience': 3,  # number of epochs to wait as monitored value does not change before decreasing LR
            'verbose': True,  # print a message when LR is changed
            'threshold': 1e-3,  # when to consider the monitored varaible not changing (focus on significant changes)
            'min_lr': 1e-7,  # lower bound on learning rate, not decreased further
            'cooldown': 0  # number of epochs to wait before resuming operation after LR was reduced
        }
    }


