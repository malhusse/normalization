# This should do everything and go from a T2w Dicom Series
# and output a normalized T2w dicom series or mha file

import os
from re import X
from statistics import linear_regression
import SimpleITK as sitk
import argparse
from norm_helpers import *
from scipy.stats import linregress

def main(input_folder, output_folder, debug_folder, debug=False):

    mrcnn_model = get_mrcnn_model()
    reader = sitk.ImageSeriesReader()
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
   
    if debug:
        os.makedirs(os.path.join(debug_folder, 'MRCNN_OUTPUT'), exist_ok=True)
        os.makedirs(os.path.join(debug_folder, 'NORM_OUTPUT'), exist_ok=True)
   
    print('Now running on {}'.format(input_folder))
   
    t2w_img, t2w_re_arr, t2w_re_no_arr = read_input(reader, input_folder)
    print("Finished resampling for {}".format(input_folder))
   
    volume_masks = feed_image_to_mrcnn(mrcnn_model, t2w_re_no_arr, debug_folder, debug)
    print("Finished MRCNN inference for {}".format(input_folder))
   
    # get the mean intensity of each contour
    true_ints_dict, true_ints, target_ints, CaseType = get_intensities(t2w_re_arr, volume_masks)
   
    # get the piecewise linear function based on the true_ints and target_ints lists
    interp_function = get_interpolation_function(true_ints, target_ints)
    if not interp_function:
        print("ERROR: No enough points to fit. No output was written.")
        return
   
    if debug:
        # draw the spline curve and save fig
        plot_spline_curve(CaseType, interp_function, true_ints_dict, os.path.join(debug_folder, 'NORM_OUTPUT'))
            
    t2w_spline_norm = interp_function(sitk.GetArrayFromImage(t2w_img)).clip(min=0)   # The .clip part is questionable (it clips out negative values)     

    # fit a linear regression to the intensities
    linereg_result = linregress(true_ints, target_ints)
    t2w_spline_norm_linreg = (sitk.GetArrayFromImage(t2w_img) * linereg_result.slope + linereg_result.intercept).clip(0)

    if debug:
        plot_linreg_curve(CaseType, linereg_result, true_ints_dict, os.path.join(debug_folder, 'NORM_OUTPUT'))

    os.makedirs(output_folder, exist_ok=True)
   
    print("Finished spline normalization for {}".format(input_folder))

    # write the output as a dicom series
    print('defaulting to MHA output')
    # output using the linear spline, uses piecewise linear function between every two points
    write_mha_file(reader, t2w_img, t2w_spline_norm, writer, output_folder, name='T2.mha')

    # output using a linear regression, fits a single linear to all points
    write_mha_file(reader, t2w_img, t2w_spline_norm_linreg, writer, output_folder, name='T2_linreg.mha')

    print("Finished writing output for series {} in path {}".format(input_folder, output_folder))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script will normalize the T2 weighted dicom series')
    parser.add_argument('--input_dir', type=str, required=False, default=os.path.abspath('INPUT'),
                        help='The path to the input directory which contains the MRI data DCM sequences')
    parser.add_argument('--output_dir', type=str, required=False, default=os.path.abspath('OUTPUT'),
                        help='The path to the output directory which normalized DCM files are output to')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.output_dir, True)