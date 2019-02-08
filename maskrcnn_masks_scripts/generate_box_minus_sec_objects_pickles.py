''' The file generates boxes for all objects based on the predictions of MASK-R-CNN and the
    true 2D boxes, subtracting secondary objects of the same class of the true label. Final
    result is a 2d mask (which is = true box - masks of the same class excep

    1. Inference using pretrained maskrcnn.
    2. IOU values between predicted masks and true 2D boxes are used to detect secondary objects in each true
        2d box of the same class as the true label.
    3. final returned box is the true 2d box subtracting from it all masks of secondary objects of the same type.


final filtered masks count:  23862
final total object count:  51865


'''

import matplotlib.pyplot as plt
from os.path import join
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import pandas as pd
import pickle

config_file = "../configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
kitti_images_path = '../../../frustum-pointnets/dataset/KITTI/object/training/image_2'
kitti_labels_path = '../../../frustum-pointnets/dataset/KITTI/object/training/label_2'


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
def get_boxes_inside_true_box_using_iou_class_label(image, true_box, true_label, boxes, labels,
                                                    iou_threshold=0.0):
    true_label = true_label.lower()


    if true_label not in ['car', 'van', 'pedestrian', 'cyclist']:
        return np.array([]), np.array([])

    idx = []
    ious = []

    labels_mappings= {'person': 'pedestrian', 'car': 'car', 'van': 'car', 'truck':'car'}

    # check which masks are contained inside the true_box
    for i in range(0, len(boxes)):
        iou = bb_intersection_over_union(true_box, boxes[i])
        if iou > iou_threshold and labels_mappings.get(labels[i].lower(), ' ') == true_label:
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


# convert a box to an image with 1's inside the box and 0's outside, dimensions of the array given by the image
def convert_box_to_mask(true_box, image):
    h, w = image.shape[:2]
    true_box_as_mask = np.zeros((h, w))

    true_box = [int(b) for b in true_box]
    x1, y1, x2, y2 = true_box

    true_box_as_mask[y1:y2, x1:x2] = 1
    return true_box_as_mask

# filter out masks of secondary objects of  the same type
def refine_main_box(indices, best_iou_index, true_box, masks, image):

    refined_box = convert_box_to_mask(true_box, image)
    # no secondary objects of the same type, return true box as a mask
    if len(indices) == 1:
        return refined_box

    # some secondary objects of the same type exist
    else:
        main_mask = masks[best_iou_index]
        masks = masks[indices]
        sum_masks = np.sum(masks, axis=0)

        refined_box = refined_box + main_mask - sum_masks
        refined_box = np.where(refined_box > 0, 1, 0)
        return refined_box


# get predictions and save masks, annotated images, predicted boxes
def get_predictions_box_level_from_whole_image_using_iou_filtering_class_sec_objects(
        path, img_index, save_path='../KITTI_predictions_true_box_filtered_mask/'):

    image = load(path, img_index, center_image=True)
    true_boxes, true_labels = load_true_boxes_from_label(kitti_labels_path, fileindex=img_index)

    final_masks = list()

    predictions_img, boxes, labels, pred_object = coco_demo.run_on_opencv_image(image)

    boxes = np.array(boxes.numpy(), dtype=np.int64)
    scores = np.array(pred_object.get_field("scores").numpy(), dtype=np.float32)
    masks = pred_object.get_field('mask')
    if len(masks.size()) > 3:
        masks = masks.squeeze(1)

    masks = np.array([m.numpy() for m in masks])
    labels = np.array(labels)

    for true_box, true_label in zip(true_boxes, true_labels):
        indicies, ious_full = get_boxes_inside_true_box_using_iou_class_label(image,
                                                                              true_box,
                                                                              true_label,
                                                                              boxes,
                                                                              labels)

        if len(indicies) <= 1:
            selected_masks = np.array([], dtype=np.int8)
        else:
            best_iou_index = np.argmax(ious_full)
            refined_box = refine_main_box(indicies, best_iou_index, true_box, masks, image)
            selected_masks = refined_box


        final_masks.append(selected_masks)

    with open(join(save_path, 'mask_arrays', 'masks_' + str(img_index).zfill(6)), 'wb') as file:
        pickle.dump(final_masks, file)

    # concatenate everything in one line to save as TXT file
    lines = [label.upper() + ' ' + ' '.join([str(x) for x in box]) for box, label in zip(boxes, labels)]

    # save TXT files and images
    # 2D Boxes
    save_filename = join(save_path, '2d_boxes', 'MaskRCNN_predictions_' + str(img_index).zfill(6) + '.txt')
    pd.DataFrame(lines).to_csv(save_filename, index=False, header=False)

    # Annoteated images
    save_filename = join(save_path, 'annotated_images', 'MaskRCNN_predictions_' + str(img_index).zfill(6) + '.jpg')
    plt.imsave(save_filename, predictions_img)

    return predictions_img, final_masks


kitti_images_count = 7481 # 7481


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

    filtered_masks = 0
    total_objects = 0

    for i in range(kitti_images_count):
        index = i

        predictions_img, masks = get_predictions_box_level_from_whole_image_using_iou_filtering_class_sec_objects(
            kitti_images_path, index)

        total_objects += len(masks)
        filtered_masks += np.sum([0 if len(m) == 0 else 1 for m in masks])

        if (i % 250 == 0):
            print("Image at Index i DONE: ", i)

    print("final filtered masks count: ", filtered_masks)
    print("final total object count: ", total_objects)