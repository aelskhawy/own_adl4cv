# extra check for the class to remove incorrect class predictions
# Add mask augmentation along with box augmentation
# Change the limit of the number of points in the mask point cloud from 10 to 50


# reads the pickle files for predicted masks and labels
# each true box can either have one mask or no masks at all



from __future__ import print_function

import os
import sys
import numpy as np
from scipy.ndimage.interpolation import shift
import cv2
from PIL import Image

MASK_PICKLES_PATH = '/home/os/Desktop/adl4cv/maskrcnn_benchmark_facebook/maskrcnn-benchmark/KITTI_predictions_true_box_filtered_mask/mask_arrays'
#CALIB_DATA_PATH = '/home/os/Desktop/adl4cv/frustum-pointnets/dataset/KITTI/object/training/calib'
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = '/media/os/DATA/# Last Semester/ADL4CV/frustum-pointnets'
#print(BASE_DIR, ROOT_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))

#import kitti_preperation.kitti_util2 as utils
import _pickle as pickle

from kitti_object2 import *
import argparse


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]]
    box2d_corners[1,:] = [box2d[2],box2d[1]]
    box2d_corners[2,:] = [box2d[2],box2d[3]]
    box2d_corners[3,:] = [box2d[0],box2d[3]]
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:], box2d_roi_inds


def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])

def get_box_from_mask(mask):
    print('points in mask', np.sum(mask))
    mask_inds = np.where(mask > 0)
    return np.min(mask_inds[1]), np.max(mask_inds[1]), np.min(mask_inds[0]), np.max(mask_inds[0])

def get_mask_inds(mask):
    print('points in mask', np.sum(mask))
    mask_inds = np.where(mask > 0)
    mean_x, mean_y = np.mean(mask_inds[1]), np.mean(mask_inds[0])
    return list(zip(mask_inds[1], mask_inds[0])), mean_x, mean_y

def random_shift_mask(mask, shift_ratio=0.1):
    xmin, xmax, ymin, ymax = get_box_from_mask(mask)
    hor_shift = (xmax - xmin) * shift_ratio * np.random.random()
    ver_shift = (ymax - ymin) * shift_ratio * np.random.random()

    xp = shift(mask, (ver_shift, hor_shift), mode='constant')
    xn = shift(mask, (-ver_shift, -hor_shift), mode='constant')
    yp = shift(mask, (-ver_shift, hor_shift), mode='constant')
    yn = shift(mask, (ver_shift, -hor_shift), mode='constant')

    x = xp + xn + mask + yn + yp

    return np.where(x >= 1, 1, 0)

def get_pc_mask_inds (mask,pc_image_coord,img_fov_inds):
    xmin, xmax, ymin, ymax = get_box_from_mask(mask)
    box_fov_inds_mask = (pc_image_coord[:, 0] < xmax) & \
                        (pc_image_coord[:, 0] >= xmin) & \
                        (pc_image_coord[:, 1] < ymax) & \
                        (pc_image_coord[:, 1] >= ymin)
    box_fov_inds_mask = box_fov_inds_mask & img_fov_inds

    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])

    # we will subset from pc_image_coor[box_fov_inds] to save the complexity of searching the whole pc
    tmp = pc_image_coord[box_fov_inds_mask]
    tmp[:, 0] = np.where(tmp[:, 0] < box2d_center[0], np.floor(tmp[:, 0]), np.ceil(tmp[:, 0]))
    tmp[:, 1] = np.where(tmp[:, 1] < box2d_center[1], np.floor(tmp[:, 1]), np.ceil(tmp[:, 1]))
    tmp = np.asarray(tmp, dtype=np.int64)
    inds_mask2d, mean_x, mean_y = get_mask_inds(mask)
    return np.array([tuple(coord) in inds_mask2d for coord in tmp]), box_fov_inds_mask, mean_x, mean_y



def extract_frustum_data(idx_filename, split, output_filename, viz=False,
                       perturb_box2d=False, augmentX=1, type_whitelist=['Car']):
    ''' Extract point clouds and corresponding annotations in frustums
        defined generated from 2D bounding boxes
        Lidar points and 3d boxes are in *rect camera* coord system
        (as that in 3d box label files)

    Input:
        idx_filename: string, each line of the file is a sample ID
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        viz: bool, whether to visualize extracted data
        perturb_box2d: bool, whether to perturb the box2d
            (used for data augmentation in train set)
        augmentX: scalar, how many augmentations to have for each 2D box.
        type_whitelist: a list of strings, object types we are interested in.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'), split)
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    id_list = [] # int number
    box2d_list = [] # [xmin,ymin,xmax,ymax]
    box3d_list = [] # (8,3) array in rect camera coord
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    label_list = [] # 1 for roi object, 0 for clutter
    type_list = [] # string e.g. Car
    heading_list = [] # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = [] # array of l,w,h
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    pos_cnt = 0
    all_cnt = 0

    obj_box_count = 0
    obj_mask_count = 0

    obj_num = 0
    ign_obj = 0

    for data_idx in data_idx_list:
        #if data_idx % 250 in range(1,10):
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        pc_velo = dataset.get_lidar(data_idx)

        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
        pc_rect[:,3] = pc_velo[:,3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:,0:3],
            calib, 0, 0, img_width, img_height, True)

        mask_path = os.path.join(MASK_PICKLES_PATH, 'masks_' + str(data_idx).zfill(6))
        with open(mask_path, 'rb') as f:
            mask_list = pickle.load(f)

        assert len(mask_list) == len(objects)

        for obj_idx in range(len(objects)):
            if objects[obj_idx].type not in type_whitelist:
                continue

            # 2D BOX: Get pts rect backprojected
            # mask 2d is a list of arrays that contains points the belongs to the mask
            mask2d = mask_list[obj_idx]
            box2d = objects[obj_idx].box2d

            for _ in range(augmentX):

                # Augment data by box2d perturbation
                if perturb_box2d:
                    xmin, ymin, xmax, ymax = random_shift_box2d(box2d)
                else:
                    xmin, ymin, xmax, ymax = box2d

                if mask2d.shape[0] != 0 and np.sum(mask2d) != 0:
                    if perturb_box2d:
                        mask = random_shift_mask(mask2d)
                        mask_fov_inds, box_fov_inds_mask, mean_x, mean_y  = \
                            get_pc_mask_inds(mask, pc_image_coord, img_fov_inds)
                    else:
                        #mask = np.array([point for point in mask2d])
                        mask_fov_inds, box_fov_inds_mask, mean_x, mean_y = \
                            get_pc_mask_inds(mask2d, pc_image_coord, img_fov_inds)

                # No mask or a sparse mask point cloud
                if mask2d.shape[0] == 0 or (np.sum(mask2d) == 0) or (np.sum(mask_fov_inds) < 25):
                    obj_box_count += 1

                    box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                                   (pc_image_coord[:, 0] >= xmin) & \
                                   (pc_image_coord[:, 1] < ymax) & \
                                   (pc_image_coord[:, 1] >= ymin)
                    box_fov_inds = box_fov_inds & img_fov_inds

                    # pc in box fov
                    pc_in_fov = pc_rect[box_fov_inds, :]
                    #Get frustum angle (according to center pixel in 2D BOX)
                    box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
                    uvdepth = np.zeros((1,3))
                    uvdepth[0,0:2] = box2d_center
                    uvdepth[0,2] = 20 # some random depth
                    box2d_center_rect = calib.project_image_to_rect(uvdepth)
                    frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2], box2d_center_rect[0,0])

                else:
                    obj_mask_count += 1

                    # pc in mask fov
                    # we will subset from the pc part inside the box, not the whole pc_rect
                    pc_in_box_fov = pc_rect[box_fov_inds_mask,:]
                    pc_in_fov = pc_in_box_fov[mask_fov_inds, :]

                    # frustum angel from mask not from the 2d box
                    mask2d_center = np.array([mean_x, mean_y])
                    uvdepth = np.zeros((1, 3))
                    uvdepth[0, 0:2] = mask2d_center
                    uvdepth[0, 2] = 20  # some random depth
                    mask2d_center_rect = calib.project_image_to_rect(uvdepth)

                    frustum_angle = -1 * np.arctan2(mask2d_center_rect[0,2], mask2d_center_rect[0,0])



                # 3D BOX: Get pts velo in 3d box
                obj = objects[obj_idx]
                box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
                _,inds = extract_pc_in_box3d(pc_in_fov, box3d_pts_3d)
                label = np.zeros((pc_in_fov.shape[0]))
                label[inds] = 1
                # Get 3D BOX heading
                heading_angle = obj.ry
                # Get 3D BOX size
                box3d_size = np.array([obj.l, obj.w, obj.h])

                # Reject too far away object or object without points
                if box2d[3] - box2d[1] < 25 or np.sum(label)==0:
                    ign_obj += 1
                    continue

                id_list.append(data_idx)
                box2d_list.append(np.array([xmin,ymin,xmax,ymax]))
                box3d_list.append(box3d_pts_3d)
                input_list.append(pc_in_fov)
                label_list.append(label)
                type_list.append(objects[obj_idx].type)
                heading_list.append(heading_angle)
                box3d_size_list.append(box3d_size)
                frustum_angle_list.append(frustum_angle)

                # collect statistics
                pos_cnt += np.sum(label)
                all_cnt += pc_in_fov.shape[0]
                obj_num += 1


    print("Obj_box_count {} ".format(obj_box_count/augmentX))
    print("Obj_mask_count {} ".format(obj_mask_count/augmentX ))
    print('label_count: {}'.format(pos_cnt))
    print('input_count: {}'.format(all_cnt))
    print('object_num: {}'.format(obj_num))
    print('ignore_num: {}'.format(ign_obj))

    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(box3d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(label_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(heading_list, fp)
        pickle.dump(box3d_size_list, fp)
        pickle.dump(frustum_angle_list, fp)

    if viz:
        import mayavi.mlab as mlab
        for i in range(10):
            p1 = input_list[i]
            seg = label_list[i]
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,0], p1[:,1], p1[:,2], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
                fgcolor=None, engine=None, size=(500, 500))
            mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
                colormap='gnuplot', scale_factor=1, figure=fig)
            raw_input()

def get_box3d_dim_statistics(idx_filename):
    ''' Collect and dump 3D bounding box statistics '''
    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/KITTI/object'))
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
        objects = dataset.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.type=='DontCare':continue
            dimension_list.append(np.array([obj.l,obj.w,obj.h]))
            type_list.append(obj.type)
            ry_list.append(obj.ry)

    with open('box3d_dimensions.pickle','wb') as fp:
        pickle.dump(type_list, fp)
        pickle.dump(dimension_list, fp)
        pickle.dump(ry_list, fp)

def read_det_file(det_filename):
    ''' Parse lines in 2D detection output files '''
    det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}
    id_list = []
    type_list = []
    prob_list = []
    box2d_list = []
    for line in open(det_filename, 'r'):
        t = line.rstrip().split(" ")
        id_list.append(int(os.path.basename(t[0]).rstrip('.png')))
        type_list.append(det_id2str[int(t[1])])
        prob_list.append(float(t[2]))
        box2d_list.append(np.array([float(t[i]) for i in range(3,7)]))
    return id_list, type_list, box2d_list, prob_list


def extract_frustum_data_rgb_detection(det_filename, split, output_filename,
                                       viz=False,
                                       type_whitelist=['Car'],
                                       img_height_threshold=25,
                                       lidar_point_threshold=5):
    ''' Extract point clouds in frustums extruded from 2D detection boxes.
        Update: Lidar points and 3d boxes are in *rect camera* coord system
            (as that in 3d box label files)

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        output_filename: string, the name for output .pickle file
        type_whitelist: a list of strings, object types we are interested in.
        img_height_threshold: int, neglect image with height lower than that.
        lidar_point_threshold: int, neglect frustum with too few points.
    Output:
        None (will write a .pickle file to the disk)
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    cache_id = -1
    cache = None

    id_list = []
    type_list = []
    box2d_list = []
    prob_list = []
    input_list = [] # channel number = 4, xyz,intensity in rect camera coord
    frustum_angle_list = [] # angle of 2d box center from pos x-axis

    for det_idx in range(len(det_id_list)):
        data_idx = det_id_list[det_idx]
        print('det idx: %d/%d, data idx: %d' % \
            (det_idx, len(det_id_list), data_idx))
        if cache_id != data_idx:
            calib = dataset.get_calibration(data_idx) # 3 by 4 matrix
            pc_velo = dataset.get_lidar(data_idx)
            pc_rect = np.zeros_like(pc_velo)
            pc_rect[:,0:3] = calib.project_velo_to_rect(pc_velo[:,0:3])
            pc_rect[:,3] = pc_velo[:,3]
            img = dataset.get_image(data_idx)
            img_height, img_width, img_channel = img.shape
            _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(\
                pc_velo[:,0:3], calib, 0, 0, img_width, img_height, True)
            cache = [calib,pc_rect,pc_image_coord,img_fov_inds]
            cache_id = data_idx
        else:
            calib,pc_rect,pc_image_coord,img_fov_inds = cache

        if det_type_list[det_idx] not in type_whitelist: continue

        # 2D BOX: Get pts rect backprojected
        xmin,ymin,xmax,ymax = det_box2d_list[det_idx]
        box_fov_inds = (pc_image_coord[:,0]<xmax) & \
            (pc_image_coord[:,0]>=xmin) & \
            (pc_image_coord[:,1]<ymax) & \
            (pc_image_coord[:,1]>=ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds,:]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin+xmax)/2.0, (ymin+ymax)/2.0])
        uvdepth = np.zeros((1,3))
        uvdepth[0,0:2] = box2d_center
        uvdepth[0,2] = 20 # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0,2],
            box2d_center_rect[0,0])

        # Pass objects that are too small
        if ymax-ymin<img_height_threshold or \
            len(pc_in_box_fov)<lidar_point_threshold:
            continue

        id_list.append(data_idx)
        type_list.append(det_type_list[det_idx])
        box2d_list.append(det_box2d_list[det_idx])
        prob_list.append(det_prob_list[det_idx])
        input_list.append(pc_in_box_fov)
        frustum_angle_list.append(frustum_angle)

    with open(output_filename,'wb') as fp:
        pickle.dump(id_list, fp)
        pickle.dump(box2d_list,fp)
        pickle.dump(input_list, fp)
        pickle.dump(type_list, fp)
        pickle.dump(frustum_angle_list, fp)
        pickle.dump(prob_list, fp)

    # if viz:
    #     import mayavi.mlab as mlab
    #     for i in range(10):
    #         p1 = input_list[i]
    #         fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
    #             fgcolor=None, engine=None, size=(500, 500))
    #         mlab.points3d(p1[:,0], p1[:,1], p1[:,2], p1[:,1], mode='point',
    #             colormap='gnuplot', scale_factor=1, figure=fig)
    #         fig = mlab.figure(figure=None, bgcolor=(0.4,0.4,0.4),
    #             fgcolor=None, engine=None, size=(500, 500))
    #         mlab.points3d(p1[:,2], -p1[:,0], -p1[:,1], seg, mode='point',
    #             colormap='gnuplot', scale_factor=1, figure=fig)
    #         raw_input()

def write_2d_rgb_detection(det_filename, split, result_dir):
    ''' Write 2D detection results for KITTI evaluation.
        Convert from Wei's format to KITTI format.

    Input:
        det_filename: string, each line is
            img_path typeid confidence xmin ymin xmax ymax
        split: string, either trianing or testing
        result_dir: string, folder path for results dumping
    Output:
        None (will write <xxx>.txt files to disk)

    Usage:
        write_2d_rgb_detection("val_det.txt", "training", "results")
    '''
    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/KITTI/object'), split)
    det_id_list, det_type_list, det_box2d_list, det_prob_list = \
        read_det_file(det_filename)
    # map from idx to list of strings, each string is a line without \n
    results = {}
    for i in range(len(det_id_list)):
        idx = det_id_list[i]
        typename = det_type_list[i]
        box2d = det_box2d_list[i]
        prob = det_prob_list[i]
        output_str = typename + " -1 -1 -10 "
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        output_str += "-1 -1 -1 -1000 -1000 -1000 -10 %f" % (prob)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo.')
    parser.add_argument('--gen_train', action='store_true', help='Generate train split frustum data with perturbed GT 2D boxes')
    parser.add_argument('--gen_val', action='store_true', help='Generate val split frustum data with GT 2D boxes')
    parser.add_argument('--gen_val_rgb_detection', action='store_true', help='Generate val split frustum data with RGB detection 2D boxes')
    parser.add_argument('--car_only', action='store_true', help='Only generate cars; otherwise cars, peds and cycs')
    args = parser.parse_args()

    # if args.demo:
    #     demo()
    #     exit()

    if args.car_only:
        type_whitelist = ['Car']
        output_prefix = 'frustum_caronly_'
    else:
        type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
        output_prefix = 'frustum_carpedcyc_'

    if args.gen_train:

        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/train.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'train.pickle'),
            viz=False, perturb_box2d=True, augmentX=5,
            type_whitelist=type_whitelist)


    if args.gen_val:
        extract_frustum_data(\
            os.path.join(BASE_DIR, 'image_sets/val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val.pickle'),
            viz=False, perturb_box2d=False, augmentX=1,
            type_whitelist=type_whitelist)

    if args.gen_val_rgb_detection:
        extract_frustum_data_rgb_detection(\
            os.path.join(BASE_DIR, 'rgb_detections/rgb_detection_val.txt'),
            'training',
            os.path.join(BASE_DIR, output_prefix+'val_rgb_detection.pickle'),
            viz=False,
            type_whitelist=type_whitelist)
