import torch
from Frustum3DLoss import Frustum3DLoss

class Frustum3DTrainableLoss(Frustum3DLoss):
    def __init__(self, num_heading_bin, num_size_cluster, endpoints, config, device='cuda'):
        super(Frustum3DTrainableLoss, self).__init__(num_heading_bin, num_size_cluster, endpoints, config)

        self.seg_weight = torch.tensor(1.0, requires_grad=True, device=device)
        self.corner_weight = torch.tensor(4.0, requires_grad=True, device=device)
        self.box_weight = torch.tensor(1.0, requires_grad=True, device=device)


    def forward(self, mask_label,
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label):

        _ = super().forward(mask_label,
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label)

        weighted_seg_loss = torch.pow(self.seg_weight, 2) * self.losses['seg_loss']
        weighted_corner_loss = torch.pow(self.corner_weight, 2) * self.losses['corner_loss']
        weighted_box_loss = torch.pow(self.box_weight, 2) * (
                                                self.losses['center_loss'] + self.losses['heading_class_loss'] +
                                                self.losses['size_class_loss'] +
                                                self.losses['heading_residual_normalized_loss'] * 20 +
                                                self.losses['size_residuals_normalized_loss'] * 20 +
                                                self.losses['stage1_center_loss'] +
                                                weighted_corner_loss)

        weighted_loss =  weighted_seg_loss + weighted_box_loss

        return weighted_loss

    def get_trainable_weights(self):
        return [self.seg_weight, self.corner_weight, self.box_weight]

    def print_loss_weights(self):
        print("Seg Loss Weight: ", torch.pow(self.seg_weight, 2).item())
        print("Corner Loss Weight: ", torch.pow(self.corner_weight, 2).item())
        print("Box Loss Weight: ", torch.pow(self.box_weight, 2).item())
