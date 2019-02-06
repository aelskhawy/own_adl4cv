''' The file generates masks for all objects based on the predictions of MASK-R-CNN and the
    true 2D boxes.

    1. Inference using pretrained maskrcnn.
    2. IOU values between predicted boxes and true 2D boxes are used to detect which mask belongs to which true
        2D Box.
    3. Cyclist true labels are handled separately, if the predicted box contains exactly two detected objects
        where one of them is a person and the other is a bicycle, they are combined as a single mask.
'''

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from os.path import join
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import pandas as pd
from datetime import datetime
import pickle

config_file = "../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
kitti_images_path = '../../../frustum-pointnets/dataset/KITTI/object/training/image_2'
kitti_labels_path = '../../../frustum-pointnets/dataset/KITTI/object/training/label_2'
kitti_images_count = 5 # 7481

# load an image
def load(path=kitti_images_path, img_index=0, center_image=False):
    filepath = join(path, str(img_index).zfill(6)+'.png')
    pil_image = Image.open(filepath).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    if (center_image):
        image -= np.mean(image, dtype=np.uint8)
    return image

# load the true 2d boxes
def load_true_boxes_from_label(base_dir, fileindex, zero_fill=6, prefix='', ext='.txt', true_box=[],
               allowed_classes=[]):
    file = join(base_dir, prefix+str(fileindex).zfill(zero_fill)+ext)
    df = pd.read_csv(file, header=-1, sep=' ')
    if (allowed_classes != []):
        df = df[df[0].isin(allowed_classes)]
    labels = df[0]
    return list(df.iloc[:, 4:8].apply(lambda row: list(row), axis=1)), list(labels)


# get masks inside the true box provided within a given iou threshold on boxes
def get_boxes_inside_true_box_using_iou(image, true_box, true_label, boxes,
                                        iou_threshold=0.5):
    true_label = true_label.lower()

    if true_label in ['car', 'van']:
        iou_threshold = 0.65
    elif true_label == 'pedestrian':
        iou_threshold = 0.4
    elif true_label == 'cyclist':
        iou_threshold = 0.15
    else:
        return np.array([]), np.array([])

    idx = []
    ious = []

    # check which masks are contained inside the true_box
    for i in range(0, len(boxes)):
        iou = bb_intersection_over_union(true_box, boxes[i])
        if iou >= iou_threshold:
            idx.append(i)
            ious.append(iou)
        else:
            ious.append(0)

    idx = np.asarray(idx)
    ious = np.asarray(ious)
    return idx, ious

# compute intersection over union for two boxes
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# combine boxes of a person and a bicycle
def combine_bicycle_person_boxes(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


# identify cyclist masks
def get_cyclists_masks(true_box, predicted_boxes, predicted_labels, predicted_masks,
                       iou_threshold=0.4):
    if (predicted_labels[0] != predicted_labels[1]):
        match_found = predicted_labels[0] == 'person' and predicted_labels[1] == 'bicycle'
        match_found = match_found or (predicted_labels[0] == 'bicycle' and predicted_labels[1] == 'person')
        if (match_found):
            combined_box = combine_bicycle_person_boxes(predicted_boxes[0],
                                                        predicted_boxes[1])
            iou = bb_intersection_over_union(true_box, combined_box)
            if (iou >= iou_threshold):
                combined_mask = predicted_masks[0] + predicted_masks[1]
                combined_mask = np.where(combined_mask > 0, 1, 0)
                return combined_box, combined_mask, iou, predicted_masks[0], predicted_masks[1]
    return [], [], 0


# get predictions and save masks, annotated images, predicted boxes
def get_predictions_box_level_from_whole_image_using_iou(path, img_index,
                                                         save_path='../KITTI_predictions_true_box_iou_cycfix/'):
    image = load(path, img_index, center_image=True)
    true_boxes, true_labels = load_true_boxes_from_label(kitti_labels_path, fileindex=img_index)

    final_masks = list()
    final_labels = list()
    final_boxes = list()
    final_scores = list()
    a, b = 0, 0

    predictions_img, boxes, labels, pred_object = coco_demo.run_on_opencv_image(image)

    boxes = np.array(boxes.numpy(), dtype=np.int64)
    scores = np.array(pred_object.get_field("scores").numpy(), dtype=np.float32)
    masks = pred_object.get_field('mask')
    if len(masks.size()) > 3:
        masks = masks.squeeze(1)

    masks = np.array([m.numpy() for m in masks])
    labels = np.array(labels)

    for true_box, true_label in zip(true_boxes, true_labels):
        indicies, ious_full = get_boxes_inside_true_box_using_iou(image,
                                                                  true_box,
                                                                  true_label,
                                                                  boxes)

        if true_label.lower() == 'cyclist':
            if len(indicies) == 2:
                selected_boxes, selected_masks, score, a, b = get_cyclists_masks(
                    true_box, boxes[indicies], labels[indicies], masks[indicies])
                selected_boxes = np.array(selected_boxes)
                selected_masks = np.array(selected_masks)
                selected_labels = 'cyclist' if len(selected_boxes) else ''
                selected_scores = score if len(selected_boxes) else 0
            else:
                selected_boxes, selected_masks, selected_labels, selected_scores = np.array([],
                                                                                            dtype=np.int8), np.array([],
                                                                                                                     dtype=np.int8), '', 0
        else:
            if (len(indicies) == 0):
                best_iou_index = np.array([])
                selected_boxes, selected_masks, selected_labels, selected_scores = np.array([],
                                                                                            dtype=np.int8), np.array([],
                                                                                                                     dtype=np.int8), '', 0
            else:
                best_iou_index = np.argmax(ious_full)
                selected_boxes = boxes[best_iou_index]
                selected_masks = masks[best_iou_index]
                selected_labels = labels[best_iou_index]
                selected_scores = scores[best_iou_index]

        final_masks.append(selected_masks)
        final_labels.append(selected_labels)
        final_boxes.append(selected_boxes)
        final_scores.append(selected_scores)

    with open(join(save_path, 'mask_arrays', 'masks_' + str(img_index).zfill(6)), 'wb') as file:
        pickle.dump(final_masks, file)
        pickle.dump(final_labels, file)
        pickle.dump(final_boxes, file)
        pickle.dump(final_scores, file)

    # concatenate everything in one line to save as TXT file
    lines = [label.upper() + ' ' + ' '.join([str(x) for x in box]) for box, label in zip(boxes, labels)]

    # save TXT files and images
    # 2D Boxes
    save_filename = join(save_path, '2d_boxes', 'MaskRCNN_predictions_' + str(img_index).zfill(6) + '.txt')
    pd.DataFrame(lines).to_csv(save_filename, index=False, header=False)

    # Annoteated images
    save_filename = join(save_path, 'annotated_images', 'MaskRCNN_predictions_' + str(img_index).zfill(6) + '.jpg')
    plt.imsave(save_filename, predictions_img)

    return predictions_img, final_boxes, final_masks, final_labels, final_scores


if __name__ == '__main__':
    #update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.5,
    )

    no_mask_count = 0
    single_mask_count = 0
    more_than_one_count = 0
    total_objects = 0

    for i in range(kitti_images_count):
        index = i
        if (i % 100 == 0):
            print("Image at Index i DONE: ", i)

        predictions_img, boxes, masks, labels, scores = get_predictions_box_level_from_whole_image_using_iou(
            kitti_images_path, index)

        zero_counts = sum([len(b) == 0 for b in boxes])
        one_counts = sum([len(b) > 0 and len(b.shape) == 1 for b in boxes])
        more_counts = len(boxes) - one_counts - zero_counts

        total_objects += len(boxes)
        single_mask_count += one_counts
        no_mask_count += zero_counts
        more_than_one_count += more_counts

    print("zero masks: ", no_mask_count)
    print("one mask count: ", single_mask_count)
    print("more masks:", more_than_one_count)
    print("total object count: ", total_objects)

