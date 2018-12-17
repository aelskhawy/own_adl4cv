import torch
import torch.nn as nn
from model_utils import conv2d_block, fc_block


class RegressionBox3DModel(nn.Module):
    def __init__(self, m_points, n_channels, n_classes, batch_norm_decay=0.9,
                 num_heading_bin=12, num_size_cluster=8):
        super(RegressionBox3DModel, self).__init__()
        self.batch_norm_decay = batch_norm_decay
        self.m_points = m_points
        self.channels = n_channels
        self.n_classes = n_classes
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster

        self.object_features = None
        self.outputs = None


        # object feature extractor
        conv_layer1 = conv2d_block(self.channels, 128, decay=self.batch_norm_decay)
        conv_layer2 = conv2d_block(128, 128, decay=self.batch_norm_decay)
        conv_layer3 = conv2d_block(128, 256, decay=self.batch_norm_decay)
        conv_layer4 = conv2d_block(256, 512, decay=self.batch_norm_decay)
        max_pool5 = nn.MaxPool2d(kernel_size=(self.m_points, 1), stride=(2,2))
        self.object_features_extractor = nn.Sequential(*conv_layer1, *conv_layer2, *conv_layer3,
                                                       *conv_layer4, max_pool5)

        # box regressor
        # number of regressors, 3 for box center in 3D
        regressors_count = 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4
        fc_layer6 = fc_block(512+self.n_classes, 512, decay=self.batch_norm_decay)
        fc_layer7 = fc_block(512, 256, decay=self.batch_norm_decay)
        fc_layer8 = fc_block(256, regressors_count, decay=self.batch_norm_decay, activation_layer=None)
        self.box_regressor = nn.Sequential(*fc_layer6, *fc_layer7, *fc_layer8)

    def forward(self, input_point_cloud: torch.Tensor, one_hot_vector: torch.Tensor):

        # (batch_size X channels X m_points X 1)
        input_point_cloud = input_point_cloud.permute(0, 2, 1).unsqueeze(3)
        self.object_features = self.object_features_extractor(input_point_cloud)

        # (B x 512)
        self.object_features = self.object_features.squeeze(3).squeeze(2)

        # (B x 512+num_classes)
        self.object_features = torch.cat((self.object_features, one_hot_vector), dim=1)

        # (B, 3 + self.num_heading_bin * 2 + self.num_size_cluster * 4)
        # The first 3 numbers: box center coordinates (cx,cy,cz),
        # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
        # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        self.outputs = self.box_regressor(self.object_features)

        return self.outputs














