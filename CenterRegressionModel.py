import torch
import torch.nn as nn
from model_utils import conv2d_block, fc_block

class CenterRegressionModel(nn.Module):
    def __init__(self, m_points, n_channels, n_classes, batch_norm_decay=0.9):
        super(CenterRegressionModel, self).__init__()
        self.batch_norm_decay = batch_norm_decay
        self.m_points = m_points
        self.channels = n_channels
        self.n_classes = n_classes

        # object features extractor
        conv_layer1 = conv2d_block(self.channels, 128, decay=self.batch_norm_decay)
        conv_layer2 = conv2d_block(128, 128, decay=self.batch_norm_decay)
        conv_layer3 = conv2d_block(128, 256, decay=self.batch_norm_decay)
        max_pool4 = nn.MaxPool2d(kernel_size=(self.m_points, 1), stride=(2, 2))
        self.object_features_extractor = nn.Sequential(*conv_layer1, *conv_layer2, *conv_layer3, max_pool4)

        # center regressor
        fc_layer5 = fc_block(256+self.n_classes, 256, decay=self.batch_norm_decay)
        fc_layer6 = fc_block(256, 128, decay=self.batch_norm_decay)
        fc_layer7 = fc_block(128, 3, decay=self.batch_norm_decay, activation_layer=None)

        self.center_regressor = nn.Sequential(*fc_layer5, *fc_layer6, *fc_layer7)

    def forward(self, input_point_cloud, one_hot_vector):
        #print("Center network input: ", input_point_cloud.size())
        input_point_cloud = input_point_cloud.permute(0, 2, 1).unsqueeze(3)
        self.object_features = self.object_features_extractor(input_point_cloud)
        self.object_features = self.object_features.squeeze(3).squeeze(2)
        self.concat_object_features = torch.cat(
            (self.object_features.view(-1, 256), one_hot_vector), dim=1)

        self.predicted_center = self.center_regressor(self.concat_object_features)

        return self.predicted_center
