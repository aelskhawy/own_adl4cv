import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from keras.utils import to_categorical

####### Global Constants #######
g_type2class={'Car':0, 'Van': 1, 'Truck':2, 'Pedestrian':3,
              'Person_sitting': 4, 'Cyclist':5, 'Tram':6, 'Misc':7}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
g_type_mean_size = {'Car': np.array([3.88311640418, 1.62856739989, 1.52563191462]),
                    'Van': np.array([5.06763659, 1.9007158, 2.20532825]),
                    'Truck': np.array([10.13586957, 2.58549199, 3.2520595]),
                    'Pedestrian': np.array([0.84422524, 0.66068622, 1.76255119]),
                    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
                    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
                    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
                    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])}

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8


def conv2d_block(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1),
                 batch_norm_layer=True, decay=0.9, affine_transform=True, activation_layer='relu'):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    layers.append(conv_layer)

    if batch_norm_layer:
        batch_norm_layer = nn.BatchNorm2d(out_channels, momentum=1 - decay, affine=affine_transform)
        layers.append(batch_norm_layer)
    if activation_layer == 'relu':
        layers.append(nn.ReLU())
    return layers


def fc_block(in_channels, out_channels, batch_norm_layer=True, decay=0.9,
             affine_transform=True, activation_layer='relu'):
    layers = []
    fc_layer = nn.Linear(in_channels, out_channels)
    layers.append(fc_layer)

    if batch_norm_layer:
        batch_norm_layer = nn.BatchNorm1d(out_channels, momentum=1 - decay, affine=affine_transform)
        layers.append(batch_norm_layer)
    if activation_layer == 'relu':
        layers.append(nn.ReLU())
    return layers


def get_heading_loss(heading_scores, heading_class_label):
    loss = F.binary_cross_entropy_with_logits(heading_scores, heading_class_label, reduction='elementwise_mean')
    return loss


def gather_object_pointcloud(point_cloud, mask, m_points, device='cuda'):
    def mask_to_indicies(mask, batch_size, n_channels):
        indices = torch.zeros((batch_size, m_points, n_channels), dtype=torch.long)
        for i in range(batch_size):
            pos_indices = np.where(mask[i, :] > 0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0:
                if len(pos_indices) > m_points:
                    choice = np.random.choice(len(pos_indices),
                                              m_points, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                                              m_points - len(pos_indices), replace=True)
                    choice = np.concatenate((np.arange(len(pos_indices)), choice))
                np.random.shuffle(choice)
                indices[i, :] = torch.from_numpy(pos_indices[choice]).unsqueeze(1).repeat(1, 4)
        return indices.to(device)

    batch_size = mask.size()[0]
    n_channels = point_cloud.size()[2]

    #indices = torch.autograd.Variable(mask_to_indicies(mask, batch_size))
    indices = mask_to_indicies(mask, batch_size, n_channels)

    object_point_cloud = torch.zeros((mask.size()[0], m_points, n_channels))
    #object_point_cloud = torch.autograd.Variable(object_point_cloud)

    # for i in range(batch_size):
    #     count = 0
    #     for j in indices[i, :, 1]:
    #         object_point_cloud[i, count, :] = point_cloud[i, j, :]
    #         count += 1

    object_point_cloud = torch.gather(point_cloud, 1, indices)

    return object_point_cloud.to(device), indices


def point_cloud_masking(point_cloud, segmentation_logits, endpoints, NUM_OBJECT_POINT=512, device='cuda'):
    #print("inside point cloud masking")

    #print("Shape of point cloud at start: ", point_cloud.size())
    #print("Shape of seg logits at start: ", segmentation_logits.size())
    batch_size = point_cloud.size()[0]
    n_points = point_cloud.size()[1]

    mask = (segmentation_logits[:, :, 0] < segmentation_logits[:, :, 1]).view(batch_size, n_points, -1)
    #print("mask shape: ", mask.size())
    mask = mask.type(torch.FloatTensor).to(device)
    mask_count = torch.sum(mask, dim=1, keepdim=True).repeat(1, 1, 3)
    #print("mask count shape: ", mask_count.size())

    #point_cloud_xyz = point_cloud[:, :, 0:3]  # BxNx3
    point_cloud_xyz = point_cloud.narrow(dim=2, start=0, length=3)
    mask_xyz_mean = torch.sum(mask.repeat(1, 1, 3) * point_cloud_xyz, dim=1, keepdim=True)  # Bx1x3
    mask = torch.squeeze(mask, dim=2)  # BxN

    endpoints['mask'] = mask

    # avoiding division by zero when predicted number of points belonging to the object is zero
    #print(torch.max(mask_count,1))
    #print(mask_xyz_mean.size())
    mask_xyz_mean = mask_xyz_mean / torch.max(mask_count, torch.Tensor([1.0]).to(device))  # Bx1x3

    point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.repeat(1, n_points, 1)

    #point_cloud_reflection = point_cloud[:, :, 3:]
    point_cloud_reflection = point_cloud.narrow(dim=2, start=3, length=1)
    point_cloud_stage1 = torch.cat((point_cloud_xyz_stage1, point_cloud_reflection), dim=-1)

    num_channels = point_cloud_stage1.size()[2]

    object_point_cloud, _ = gather_object_pointcloud(point_cloud_stage1, mask, NUM_OBJECT_POINT, device)
    object_point_cloud = object_point_cloud.view(batch_size, NUM_OBJECT_POINT, num_channels)

    return object_point_cloud, torch.squeeze(mask_xyz_mean, dim=1), endpoints


def parse_3dregression_model_output(output, endpoints, num_heading_bin, num_size_cluster, device='cuda'):
    batch_size = output.size(0)
    center = output[:, :3]
    endpoints['center_boxnet'] = center

    # First 3 values are the center, followed by num_heading bin values
    # followed by num_heading_bin residuals
    # followed by num_size_cluster value for size scores
    # followed by num_size_cluster value for size residuals
    heading_scores = output[:, 3: 3 + num_heading_bin]
    heading_residuals_normalized = output[:, 3 + num_heading_bin: 3 + 2 * num_heading_bin]

    # BxNUM_HEADING_BIN
    endpoints['heading_scores'] = heading_scores

    # BxNUM_HEADING_BIN
    # normalized residuals (-1 to 1)
    endpoints['heading_residuals_normalized'] = heading_residuals_normalized
    
    # BxNUM_HEADING_BIN
    #De-normalize the heading residuals
    endpoints['heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin)

    size_scores_start = 3 + num_heading_bin * 2
    size_scores_end = size_scores_start + num_size_cluster
    # print(output.requires_grad) it prints true, however we get an error and to solve it i wrapped 
    # the size_scores in a Variable and added requires_grad = True
    # This fix needs to be checked conceptually 
    #https://discuss.pytorch.org/t/why-in-place-operations-on-variable-data-has-no-effects-on-backward/14444
    # BxNUM_SIZE_CLUSTER
    size_scores = output[:, size_scores_start:size_scores_end]
    #Variable(torch.ones(output.size(0),size_scores_end -size_scores_start).to('cuda'), requires_grad = True)

    size_scores_residuals_start = size_scores_end
    size_scores_residuals_end = size_scores_residuals_start + 3 * num_size_cluster
    size_residuals_normalized = output[:, size_scores_residuals_start: size_scores_residuals_end]

    # BxNUM_SIZE_CLUSTERx3
    size_residuals_normalized = size_residuals_normalized.view(-1, num_size_cluster, 3)
    endpoints['size_scores'] = size_scores
    endpoints['size_residuals_normalized'] = size_residuals_normalized

    #Expanding dim for broadcasting
    g_mean_size_arr = torch.Tensor(get_mean_size_array(num_size_cluster)).to(device).unsqueeze(0)
    size_residuals = size_residuals_normalized * g_mean_size_arr
    endpoints['size_residuals'] = size_residuals
    return endpoints


def get_mean_size_array(num_size_cluster):
    g_mean_size_arr = np.zeros((num_size_cluster, 3))  # size clustrs
    for i in range(num_size_cluster):
        g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

    return g_mean_size_arr


def convert_to_one_hot(y, num_classes, device='cuda'):
    return torch.Tensor(to_categorical(y, num_classes)).to(device)


def get_box3d_corners(center, heading_resiudals, size_residuals, num_heading_bin, num_size_cluster, mean_size_arr,
                      device='cuda'):
    ''' Disclaimer: implemented by Skhawy late on a Saturday night, so use with great caution
    Input:
        center : (B,3)
        heading_residuals : (B, NH)
        size_residuals    : (B, NS,3)
        num_heading_bin : scalar
        mean_size_arr  : (NS, 3)
    outputs:
        box3d_corners : (B, NH, NS, 8, 3)
    '''
    batch_size = center.size(0)

    # size (NH,)
    heading_bin_centers = torch.Tensor(np.arange(0, 2*np.pi, 2*np.pi/num_heading_bin)).to(device)

    # size (B,NH) + (1,NH) = (B,NH)
    headings = heading_resiudals + heading_bin_centers.unsqueeze(0)

    # (B, NS, 3) # check his code, he says B,NS,1 but it can't be
    # to check if needed in gradients or not 
    mean_sizes = mean_size_arr.unsqueeze(0) + size_residuals

    #note size_residuals added twice, once in mean_sizes and one explicit
    # (B, NS, 3)
    sizes = mean_sizes + size_residuals

    # (B, NH, NS, 3)
    sizes = sizes.unsqueeze(1).repeat(1, num_heading_bin, 1, 1)

    # (B, NH, NS)
    headings = headings.unsqueeze(2).repeat(1, 1, num_size_cluster)

    # (B, NH, NS, 3)
    centers = center.unsqueeze(1).unsqueeze(1).repeat(1, num_heading_bin, num_size_cluster, 1)

    N = batch_size * num_heading_bin * num_size_cluster

    corners_3d = get_box3d_corners_helper(centers.view(N, 3), headings.view(N), sizes.view(N, 3), device)

    return corners_3d.view(batch_size, num_heading_bin, num_size_cluster, 8, 3)


def get_box3d_corners_helper(centers_labels, headings_labels, sizes_labels, device='cuda'):
   """Input: (N,3), (N,), (N,3), Output: (N,8,3) """
   N = centers_labels.size(0)
   l = sizes_labels[:, 0].unsqueeze(1) # (N,1)
   w = sizes_labels[:, 1].unsqueeze(1) # (N,1)
   h = sizes_labels[:, 2].unsqueeze(1) # (N,1)

   x_corners = torch.cat((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), dim=1) # (N,8)
   y_corners = torch.cat((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), dim=1) # (N,8)
   z_corners = torch.cat((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), dim=1) # (N,8)
   corners = torch.cat((x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)), dim=1) # (N,3,8)
   
   # (N,) 
   c = torch.cos(headings_labels)
   # (N,)
   s = torch.sin(headings_labels)

   ones = torch.ones([N], dtype=torch.float32).to(device)
   zeros = torch.zeros([N], dtype=torch.float32).to(device)
   row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
   row2 = torch.stack([zeros,ones,zeros], dim=1)
   row3 = torch.stack([-s,zeros,c], dim=1)
   R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], dim =1 ) # Nx3x3

   corners_3d = torch.matmul(R, corners) # (N,3,8)
   corners_3d = corners_3d + centers_labels.unsqueeze(2).repeat(1, 1, 8) # (N,3,8)
   corners_3d = torch.transpose(corners_3d, dim0=1, dim1=2) # (N,8,3)

   return corners_3d


def save_checkpoint(save_file_path, model, epoch, optimizer, loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, save_file_path)
    return True


def load_checkpoint(file_path, model, optimizer, device='cuda'):
    checkpoint = torch.load(file_path)
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model.to(device), optimizer, start_epoch, loss