from torch.optim import SGD

# Config
class Config:
    NUM_HEADING_BIN = 12
    NUM_SIZE_CLUSTER = 8
    NUM_POINT = 2048
    NUM_CHANNELS = 4
    NUM_OBJECT_POINT = 512
    MOMENTUM = 0.9
    DECAY_STEP = 200000
    #DECAY_RATE = 0.7

    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_STEP = float(DECAY_STEP)
    BN_DECAY_CLIP = 0.99

    BATCH_SIZE = 32

    train_control = {
        'optimizer': SGD,  # Adam, SGD
        'optimizer_params': {'lr': 1e-3},
        'decay_steps': 200000,
        'decay_rate': 0.7,
        'lr_clip': 0.00001,
        'init_bn_decay': 0.5,
        'bn_decay_rate': 0.5,
        'bn_decay_step': 200000.0,
        'bn_decay_clip': 0.99,
        'lr_scheduler_type': 'step',  # 'exp', 'step', 'plateau', 'none'

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


