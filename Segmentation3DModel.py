import torch
import torch.nn as nn
from model_utils import conv2d_block

class Segmentation3DModel(nn.Module):
    def __init__(self, n_points, n_channels, n_classes, batch_norm_decay=0.9):
        super(Segmentation3DModel, self).__init__()
        self.batch_norm_decay = batch_norm_decay
        self.n_points = n_points
        self.channels = n_channels
        self.n_classes = n_classes

        # point feature extractor
        conv_layer1 = conv2d_block(self.channels, 64, decay=self.batch_norm_decay)
        conv_layer2 = conv2d_block(64, 64, decay=self.batch_norm_decay)
        conv_layer3 = conv2d_block(64, 64, decay=self.batch_norm_decay)
        self.point_features_extractor = nn.Sequential(*conv_layer1, *conv_layer2, *conv_layer3)

        # global feature extractor
        conv_layer4 = conv2d_block(64, 128, decay=self.batch_norm_decay)
        conv_layer5 = conv2d_block(128, 1024, decay=self.batch_norm_decay)
        max_pool6 = nn.MaxPool2d(kernel_size=(self.n_points, 1), stride=(2, 2))
        self.global_features_extractor = nn.Sequential(*conv_layer4, *conv_layer5, max_pool6)

        # regression part
        conv_layer7 = conv2d_block(1024 + 64 + self.n_classes, 512, decay=self.batch_norm_decay)
        conv_layer8 = conv2d_block(512, 256, decay=self.batch_norm_decay)
        conv_layer9 = conv2d_block(256, 128, decay=self.batch_norm_decay)
        conv_layer10 = conv2d_block(128, 128, decay=self.batch_norm_decay)
        dropout_layer11 = nn.Dropout(p=0.5)
        conv_layer12 = nn.Conv2d(128, 2, (1, 1), (1, 1))

        self.regression_network = nn.Sequential(*conv_layer7, *conv_layer8, *conv_layer9, *conv_layer10,
                                                dropout_layer11,
                                                conv_layer12)
        # initialize of weights to be checked

    def forward(self, input_point_cloud: torch.Tensor, one_hot_vector: torch.Tensor):

        # input (batch_size X n_points X channels)
        # view (batch_size X channels X n_points X 1)
        input_point_cloud = input_point_cloud.permute(0, 2, 1).unsqueeze(3)
        self.point_features = self.point_features_extractor(input_point_cloud)

        self.gloabl_features = self.global_features_extractor(self.point_features)

        # reshape one hot vector for concatenation
        one_hot_vector = one_hot_vector.unsqueeze(2).unsqueeze(2)

        self.global_plus_one_hot = torch.cat((self.gloabl_features, one_hot_vector), dim=1).repeat(1, 1, self.n_points, 1)

        self.concatenated_features = torch.cat((self.point_features, self.global_plus_one_hot), dim=1)

        self.logits = self.regression_network(self.concatenated_features)

        self.logits = self.logits.squeeze(dim=3).permute(0, 2, 1)

        return self.logits


