from Trainer import ModelTrainer
from provider import FrustumDataset
from Frustum3DModel import Frustum3DModel
from config import Config
# ------------------------------------------------

TRAIN_DATASET = FrustumDataset(npoints=Config.NUM_POINT, split='train', rotate_to_center=True,
                               random_flip=True, random_shift=True, one_hot=True)
TEST_DATASET = FrustumDataset(npoints=Config.NUM_POINT, split='val', rotate_to_center=True,
                              one_hot=True)
model = Frustum3DModel(n_points=Config.NUM_POINT,
                       m_points=Config.NUM_OBJECT_POINT,
                       n_channels=4,
                       n_classes=3)

trainer = ModelTrainer(model, TRAIN_DATASET, TEST_DATASET, Config, log_interval=10)

trainer.train(150)


