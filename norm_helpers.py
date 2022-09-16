import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import trim_mean
import cv2
import time

# Root directory of the project
# sys.path.append('MaskRCNN')  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

import numpy as np
from PIL import Image
import os
import csv
import nrrd
import re
from collections import Counter

###################################
## Image Helpers ##################
###################################
def read_input(sitk_reader, input_folder):
    t2w_dicoms = sitk_reader.GetGDCMSeriesFileNames(input_folder)
        
    sitk_reader.SetFileNames(t2w_dicoms)
    sitk_reader.MetaDataDictionaryArrayUpdateOn()
    sitk_reader.LoadPrivateTagsOn()

    t2w_img = sitk_reader.Execute()
    t2w_re_arr, t2w_re_no_arr = resample_input(t2w_img)
        
    return t2w_img, t2w_re_arr, t2w_re_no_arr

def resample_input(input_image):
    input_image_f = sitk.Cast(input_image,sitk.sitkFloat32)

    # resample image to 0.5 * 0.5, keep z spacing from before
    ref_physical_size = np.array([x*y for x,y in zip(input_image_f.GetSize(), input_image_f.GetSpacing())])
    ref_spacing = [0.5, 0.5, input_image_f.GetSpacing()[2]]
    ref_size = [(int(phys / sp)) for phys, sp in zip(ref_physical_size, ref_spacing)]

    ref_image = sitk.Image(ref_size, 2)
    ref_image.SetOrigin(input_image_f.GetOrigin())
    ref_image.SetSpacing(ref_spacing)
    ref_image.SetDirection(input_image_f.GetDirection())
    
    image_re = sitk.Resample(input_image_f, ref_image, sitk.Transform(),
                                    sitk.sitkLinear, 0.0, input_image_f.GetPixelID())

    image_re_no = normalize(image_re)

    image_re_arr = sitk.GetArrayFromImage(image_re)
    image_re_no_arr = sitk.GetArrayFromImage(image_re_no)

    return image_re_arr, image_re_no_arr

def normalize(input_image):
    # Gets the value of the specified percentiles
    array = np.ndarray.flatten(sitk.GetArrayFromImage(input_image))
    lowerperc = np.percentile(array, 2)  # 2
    upperperc = np.percentile(array, 98)  # 98

    intensity_transform = sitk.IntensityWindowingImageFilter()
    intensity_transform.SetWindowMinimum(lowerperc)
    intensity_transform.SetWindowMaximum(upperperc)
    intensity_transform.SetOutputMinimum(0.0)
    intensity_transform.SetOutputMaximum(255.0)
    final_image = sitk.Cast(intensity_transform.Execute(input_image), sitk.sitkFloat32)

    return final_image


def write_dicom_series(sitk_reader, orig_img, mod_img_arr, sitk_writer, output_path):
    # this function will take the required components
    # needed to write the normalized image as a DICOM
    # series
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = orig_img.GetDirection()
    series_tag_values = [
                        (k, sitk_reader.GetMetaData(0, k))
                        for k in tags_to_copy
                        if sitk_reader.HasMetaDataKey(0, k)] + \
                    [("0008|0031", modification_time),  # Series Time
                     ("0008|0021", modification_date),  # Series Date
                     ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                     ("0020|000e", "1.2.826.0.1.3680043.2.1125." +
                      modification_date + ".1" + modification_time),
                     # Series Instance UID
                     ("0020|0037",
                      '\\'.join(map(str, (direction[0], direction[3],
                                          direction[6],
                                          # Image Orientation (Patient)
                                          direction[1], direction[4],
                                          direction[7])))),
                     ("0008|103e",
                      sitk_reader.GetMetaData(0, "0008|103e")
                      # Series Description
                      + " Processed-Spline-Normalized")]
    
    mod_img = sitk.Cast(sitk.GetImageFromArray(mod_img_arr), sitk.sitkUInt16)
    mod_img.SetOrigin(orig_img.GetOrigin())
    mod_img.SetSpacing(orig_img.GetSpacing())

    for i in range(mod_img.GetDepth()):
        image_slice = mod_img[:,:,i]
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        image_slice.SetOrigin(orig_img.GetOrigin())
        # Slice specific tags.
        #   Instance Creation Date
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        #   Image Position (Patient)
        image_slice.SetMetaData("0020|0032", '\\'.join(
            map(str, mod_img.TransformIndexToPhysicalPoint((0, 0, i)))))
        #   Instace Number
        image_slice.SetMetaData("0020|0013", str(i))
        
        sitk_writer.SetFileName(os.path.join(output_path, str(i) + '.dcm'))
        sitk_writer.Execute(image_slice)

def write_mha_file(sitk_reader, orig_img, mod_img_arr, sitk_writer, output_path, name='T2.mha'):
    mod_img = sitk.Cast(sitk.GetImageFromArray(mod_img_arr), sitk.sitkUInt16)
    mod_img.SetOrigin(orig_img.GetOrigin())
    mod_img.SetSpacing(orig_img.GetSpacing())
    mod_img.SetDirection(orig_img.GetDirection())
    
    # Write Normalized image
    sitk_writer.SetFileName(os.path.join(output_path, name))
    sitk_writer.Execute(mod_img)

    # Write Original image
    #sitk_writer.SetFileName(os.path.join(output_path, 'T2-Original.mha'))
    #sitk_writer.Execute(orig_img)

###################################
## Normalization Helpers ##########
###################################

def get_intensities(input_image_arr, volume_masks):
    true_ints_dict = {}
    for class_id, class_label in class_legend.items():
        # The pixels that you get the mean for, has a 1D array, have to be ordered first, then 10 percent of the values, 
        # at the top and at the bottom, will be clipped out first before caluculating the mean.
        class_mean_int = trim_mean(input_image_arr[volume_masks[:,class_index_map[class_id], :, :] > 0], 0.10) # 10%

        if not np.isnan(class_mean_int):
            true_ints_dict[class_label] = class_mean_int

    # true_ints, target_ints = [0], [0]
    true_ints, target_ints = [], []


    G = true_ints_dict['GM']
    F = true_ints_dict['Femur']
    B = true_ints_dict['Bladder']

    # Case 1; Correct case
    CaseType = 1
    #if (G < F < B):
        # true_ints_dict.pop('Femur')

    # Case 2: GBF -> Remove Femur
    if (G < B < F):
        true_ints_dict.pop('Femur')
        CaseType = 2

    # Case 3: FGB -> Remove Femur
    if (F < G < B):
        true_ints_dict.pop('Femur')
        CaseType = 3

    # Case 4: FBG -> Remove GM
    if (F < B < G):
        true_ints_dict.pop('GM')
        CaseType = 4

    # Case 5: BFG -> Error
    if (B < F < G): # This is an error
        true_ints_dict.pop('Bladder')
        true_ints_dict.pop('GM')
        CaseType = 5
        
    # Case 6
    if (B < G < F):
        true_ints_dict.pop('Bladder')
        CaseType = 6


    # Step: Convert values to a list
    for label, label_int in true_ints_dict.items():
        true_ints.append(label_int)
        target_ints.append(global_intensities[label])

    return true_ints_dict, true_ints, target_ints, CaseType

def plot_spline_curve(CaseType, interp_function, true_ints_dict, output_folder=None, min_value=0):
    
    shifted_ints_dict = {}
    
    for label, value in true_ints_dict.items():
        shifted_ints_dict[label] = interp_function(value)

    for name in true_ints_dict.keys():
        plt.scatter(true_ints_dict[name], shifted_ints_dict[name], label=name)

    # interpolate the curve across the intensity spectrum
    # x_min = min(true_ints_dict.values())
    x_min = min_value
    x_max = max(true_ints_dict.values())*1.2
    #  Create the plotting axes of the plot.
    x_vals = np.arange(x_min, x_max, .1)
    y_vals = interp_function(x_vals).clip(min=0)
    plt.plot(x_vals, y_vals)
    # Add labels, clean up, make pretty, and save.
    plt.xlabel('T2 Intensities original', fontsize=20)
    plt.ylabel('T2 Intensities new', fontsize=20)
    plt.legend(loc='best')
    
    if output_folder:
        fileName = 'spline_curve_CaseType' + str(CaseType) + '.png'
        plt.savefig(os.path.join(output_folder, fileName), dpi=320)
    plt.close()

def plot_linreg_curve(CaseType, linereg_result, true_ints_dict, output_folder=None, min_value=0):
    
    shifted_ints_dict = {}
    
    for label, value in true_ints_dict.items():
        shifted_ints_dict[label] = value * linereg_result.slope + linereg_result.intercept

    for name in true_ints_dict.keys():
        plt.scatter(true_ints_dict[name], shifted_ints_dict[name], label=name)

    # interpolate the curve across the intensity spectrum
    # x_min = min(true_ints_dict.values())
    x_min = min_value
    x_max = max(true_ints_dict.values())*1.2
    #  Create the plotting axes of the plot.
    x_vals = np.arange(x_min, x_max, .1)
    y_vals = (x_vals * linereg_result.slope + linereg_result.intercet).clip(min=0)
    plt.plot(x_vals, y_vals)
    # Add labels, clean up, make pretty, and save.
    plt.xlabel('T2 Intensities original', fontsize=20)
    plt.ylabel('T2 Intensities new', fontsize=20)
    plt.legend(loc='best')
    
    if output_folder:
        fileName = 'linreg_curve_CaseType' + str(CaseType) + '.png'
        plt.savefig(os.path.join(output_folder, fileName), dpi=320)
    plt.close()

def get_interpolation_function(true_ints, target_ints):
    # depending on the number of elements in true_ints that
    # we use either a linear or quadratic interpolation
    # the cubic spline interpolation doesn't work unless
    # measured bladder > femur, which is not always the case

    # len(true_ints) should equal how many structures were detected.
    
    # Step: Add Cubic spline here, but how will we do it without having 4 point ?

    # If we have 3 points use a quadratic spline
    # if len(true_ints) > 2:
    #     return interpolate.interp1d(true_ints, target_ints, kind='quadratic', fill_value='extrapolate', assume_sorted=True)

    # if we have at least two points then use linear spline
    if len(true_ints) > 1:
        return interpolate.interp1d(true_ints, target_ints, kind='linear', fill_value='extrapolate', assume_sorted=True)

    # If we have less than two detected structures skip the patient
    else:
        print('Not enough detected structures to interpolate.')
        return interpolate.interp1d(true_ints, target_ints, kind='zero', fill_value='extrapolate', assume_sorted=True)

###################################
## Mask-RCNN Helpers ##############
###################################
class NormConfig(Config):

    NAME = "norm"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + Organs

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 646
    VALIDATION_STEPS =  10

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.9

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200 #200 original

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 10 #200 original

class NormInferenceConfig(NormConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


###################################
## Mask-RCNN Helper Functions #####
###################################
def get_mrcnn_model():
    config = NormInferenceConfig()
    weights_path = 'mask_rcnn_norm_0040_1.h5'
    mrcnn_model = modellib.MaskRCNN(mode='inference', config=config,
                              model_dir = os.path.join('MaskRCNN', 'logs'))

    mrcnn_model.load_weights(weights_path, by_name=True)

    return mrcnn_model

def feed_image_to_mrcnn(mrcnn_model, input_img_arr, debug_folder, debug):
    volume_masks = np.zeros((input_img_arr.shape[0], len(class_legend)+1, input_img_arr.shape[1], input_img_arr.shape[2]))
    
    for i, t2w_slice in enumerate(input_img_arr):

            mrcnn_output = mrcnn_model.detect([np.stack((t2w_slice,)*3, axis=-1)], verbose=0)[0]
            mrcnn_output = filter_predictions(mrcnn_output)

            if mrcnn_output is None:
                continue

            # now we need to merge the mrcnn outputs based on class_ids
            t2w_slice_masks = np.zeros((len(class_legend)+1, t2w_slice.shape[0], t2w_slice.shape[1]))
            
            roi_ids = mrcnn_output['class_ids']
            roi_masks = np.transpose(mrcnn_output['masks'],(2,0,1)).astype(np.uint8)

            for roi_id, roi_mask in zip(roi_ids, roi_masks):
                if roi_id in class_legend.keys():
                    t2w_slice_masks[class_index_map[roi_id], :, :] += roi_mask

            t2w_slice_masks = np.clip(t2w_slice_masks, 0, 1)

            if debug:
                if np.any(t2w_slice_masks):
                    # TODO: lower the opacity of the overlayed masks (~alpha=.4?)
                    visualize.display_instances(np.stack((t2w_slice,)*3, axis=-1), mrcnn_output['rois'], mrcnn_output['masks'],
                                                mrcnn_output['class_ids'], class_names, mrcnn_output['scores'])

                    
                    plt.savefig(os.path.join(debug_folder, 'MRCNN_OUTPUT', 'OVERLAY_SLICE_{}.png'.format(i)), dpi=320)
                    plt.close()

            volume_masks[i] = t2w_slice_masks

    # save the masks as nrrd files, might need GM for HBV later
        
    if debug:
        for i in range(1, volume_masks.shape[1]):
            nrrd.write(os.path.join(debug_folder, 'MRCNN_OUTPUT', '{}.nrrd'.format(array_idx_legend[i])), volume_masks[:,i,:,:])

    return volume_masks

# Filter detected structures based on confidence score,
# if more than one detection, keep only one with highest score
# use class-based confidence score thresholding
threshold_dict = {
    # 6: .94, # GM
    6: .92, # GM
    4: .997, # bladder -- might need to push this to .998 for some cases..
    3: .91 # Femur
}

def filter_predictions(pred_dict):
#     output_
    roi_ids = pred_dict['class_ids']
    roi_scores = pred_dict['scores']
    roi_boxes = pred_dict['rois']
    roi_masks = np.transpose(pred_dict['masks'], (2,0,1))
    
    filtered_ids = []
    filtered_scores = []
    filtered_boxes = []
    filtered_masks = []
    
    for roi_id, roi_score, roi_box, roi_mask in zip(roi_ids, roi_scores, roi_boxes, roi_masks):
        if roi_id in class_legend.keys():
            if roi_score >= threshold_dict[roi_id]:
                # if the structure is already in the filtered set
                # check which one has the higher score and keep it
                if roi_id in filtered_ids:
                    # get the index of the already present roi
                    index = filtered_ids.index(roi_id)
                    # if the new roi has higher score..
                    if roi_score > filtered_scores[index]:
                        # remove the old roi
                        filtered_ids.pop(index)
                        filtered_scores.pop(index)
                        filtered_boxes.pop(index)
                        filtered_masks.pop(index)
                    # if the existing roi has higher score, skip this roi
                    else:
                        continue

                # if the roi isn't already picked or the old was removed
                # then add the new roi info
                filtered_ids.append(roi_id)
                filtered_scores.append(roi_score)
                filtered_boxes.append(roi_box)
                filtered_masks.append(roi_mask)
    
    if filtered_ids:
        filtered_results = {}
        filtered_results['class_ids'] = np.asarray(filtered_ids)
        filtered_results['scores'] = np.asarray(filtered_scores)
        filtered_results['rois'] = np.asarray(filtered_boxes)
        filtered_results['masks'] = np.transpose(np.asarray(filtered_masks), (1,2,0))
        return filtered_results

    else:
        return None

###################################
## Constants ######################
###################################

class_legend = {
    6: "GM",
    3: "Femur",
    4: "Bladder",
    }

# save memory and skip unused rois
class_index_map ={
    6: 1,
    3: 2,
    4: 3
}

array_idx_legend = {
    0: 'BKG',
    1: 'GM',
    2: 'Femur',
    3: 'Bladder'
}

global_intensities= {
    'GM': 329,
    'Femur': 663,
    'Bladder': 984,
    }

class_names = [
    'BG', 'Prostate', 'Muscle',
    'Femur', 'Bladder', 'PZ', 'GM'
]

tags_to_copy = ["0010|0010",  # Patient Name
                "0010|0020",  # Patient ID
                "0010|0030",  # Patient Birth Date
                "0020|000D",  # Study Instance UID, for machine consumption
                "0020|0010",  # Study ID, for human consumption
                "0008|0020",  # Study Date
                "0008|0030",  # Study Time
                "0008|0050",  # Accession Number
                "0008|0060",  # Modality
                "0028|0100",  # bits allocated
                "0028|0101",  # bits stored
                "0028|0102",  # high bit
                "0028|0103"   # pixel representation
                ]
