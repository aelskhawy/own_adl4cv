from Frustum3DLoss import Frustum3DLoss
from provider import FrustumDataset
from torch.optim import Adam
from Frustum3DModel import Frustum3DModel
from Trainer import ModelTrainer
from Box3DModelPipeline import Box3DModelBlock, Box3DModelTrainer, Box3DModelLoss
from Segmentation3DModelPipeline import Segmentation3DModelBlock,\
    Segmentation3DModelTrainer, Segmentation3DModelLoss


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

    loss_function = Segmentation3DModelLoss

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
}



TRAIN_DATASET = FrustumDataset(npoints=Config.NUM_POINT, split='train', rotate_to_center=True,
                               random_flip=True, random_shift=True, one_hot=True, resample_method=Config.resampling_method)
TEST_DATASET = FrustumDataset(npoints=Config.NUM_POINT, split='val', rotate_to_center=True,
                              one_hot=True, resample_method=Config.resampling_method)

seg_model = Segmentation3DModelBlock(n_points=Config.NUM_POINT, n_channels=4, n_classes=3)
seg_trainer = Segmentation3DModelTrainer(seg_model, TRAIN_DATASET, TEST_DATASET, Config, log_interval=10)

seg_trainer.train(10)

Config.loss_function = Box3DModelLoss
box_model = Box3DModelBlock(Config.NUM_POINT, Config.NUM_OBJECT_POINT, n_channels=Config.NUM_CHANNELS,
                       n_classes=3, resample_method=Config.resampling_method)
box_trainer = Box3DModelTrainer(box_model, TRAIN_DATASET, TEST_DATASET, Config, log_interval=10)

box_trainer.train(10)

model = Frustum3DModel(Config.NUM_POINT, Config.NUM_OBJECT_POINT, Config.NUM_CHANNELS, 3)
model.segmentation_model.load_state_dict(seg_trainer.best_model.state_dict())
model.center_regression_model.load_state_dict(box_trainer.best_model.center_regression_model.state_dict())
model.regression_box3d_model.load_state_dict(box_trainer.best_model.regression_box3d_model.state_dict())

Config.loss_function = Frustum3DLoss
Config.variable_loss_weights = False
Config.train_control['optimizer_params']['lr'] = 5e-4

trainer = ModelTrainer(model, TRAIN_DATASET, TEST_DATASET, Config, log_interval=10)
trainer.train(10)