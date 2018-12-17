import torch
import torch.nn as nn
from Segmentation3DModel import Segmentation3DModel
from CenterRegressionModel import CenterRegressionModel
from RegressionBox3DModel import RegressionBox3DModel
from model_utils import point_cloud_masking, parse_3dregression_model_output
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d


class Frustum3DModel(nn.Module):
    def __init__(self, n_points, m_points, n_channels, n_classes, device='cuda', batch_norm_decay=0.9,
                 num_heading_bin=12, num_size_cluster=8):
        super(Frustum3DModel, self).__init__()

        self.device = device
        self.m_points = m_points
        self.mask_mean_xyz = None


        self.endpoints = {}

        self.segmentation_model = Segmentation3DModel(n_points, n_channels=n_channels, n_classes=n_classes,
                                                      batch_norm_decay=batch_norm_decay)

        self.center_regression_model = CenterRegressionModel(m_points, n_channels=n_channels, n_classes=n_classes,
                                                             batch_norm_decay=batch_norm_decay)

        self.regression_box3d_model = RegressionBox3DModel(m_points, n_channels=n_channels, n_classes=n_classes,
                                                           batch_norm_decay=batch_norm_decay,
                                                           num_heading_bin=num_heading_bin,
                                                           num_size_cluster=num_size_cluster)

        self.num_heading_bin = self.regression_box3d_model.num_heading_bin
        self.num_size_cluster = self.regression_box3d_model.num_size_cluster


    def forward(self, input_point_cloud, one_hot_vector):
        self.segmentation_logits = self.segmentation_model(input_point_cloud, one_hot_vector)
        self.endpoints['mask_logits'] = self.segmentation_logits
        #print("Endpoints after seg model: ", type(self.endpoints))

        self.object_point_cloud, self.mask_mean_xyz, self.endpoints = point_cloud_masking(input_point_cloud,
                                                                                          self.segmentation_logits,
                                                                                          self.endpoints,
                                                                                          self.m_points)

        # to check if we need to change to variable here
        #self.object_point_cloud = torch.autograd.Variable(self.object_point_cloud)

        #print('input to center net:', self.object_point_cloud.size())
        self.predicted_center_delta = self.center_regression_model(self.object_point_cloud, one_hot_vector)

        #print("Endpoints after center model: ", type(self.endpoints))

        self.endpoints['stage1_center'] = self.predicted_center_delta + self.mask_mean_xyz

        self.object_point_cloud[:, :, 0:3] =self.object_point_cloud[:, :, 0:3] - self.predicted_center_delta.unsqueeze(1)

        self.boxmodel_output = self.regression_box3d_model(self.object_point_cloud, one_hot_vector)
        #print("Endpoints after box model: ", type(self.endpoints))

        self.endpoints = parse_3dregression_model_output(self.boxmodel_output, self.endpoints, self.num_heading_bin,
                                                         self.num_size_cluster, self.device)

        #print("Endpoints after parsing: ", type(self.endpoints))

        self.endpoints['center'] = self.endpoints['center_boxnet'] + self.endpoints['stage1_center']

        return self.endpoints

    def update_bn_decay(self, current_bn_decay):
        for module in self.modules():
            if isinstance(module, BatchNorm2d) or isinstance(module, BatchNorm1d):
                module.momentum = 1 - current_bn_decay



#pc_input = torch.Tensor(32, 100, 4)
#hot_vector = torch.Tensor(32, 3)
#model = Frustum3DModel(100, 50, 4, 3)
#output =model(pc_input, hot_vector)
#print(output.keys())

#model = Frustum3DModel(1024, 512, 4, 3)
#print('-' * 80)
#print([type(child) for child in model.modules()])



