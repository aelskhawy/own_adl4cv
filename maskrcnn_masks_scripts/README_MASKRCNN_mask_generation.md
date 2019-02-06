## Scripts for Masks generation using MaskRCNN

1. Go to Mask-R-CNN repository by facebook and follow the setup installation instructions here: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md

2. Copy the files in this directory and place them under `maskrcnn-benchmark/demo/`, agree to replace the `predictor.py` file.

3. Scripts can be called directory `python script_name.py`

4. All KITTI original training files should be downloaded and placed in the following relative path: `'../../../frustum-pointnets/dataset/KITTI/object/training/`.

5. Generated files are masks saved in pickle files, 2D boxes saved in TXT files, and annotated images with the predictions (3 files for each image). Files will be saved under the path: `../KITTI_predictions_true_box_iou_cycfix/`.
 