# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Wed Feb 15, 2023
------------------------------------------------------------------------------------------------------------------------
Run locally or on lopevi using labscript conda environment.

This code is used to sort the traces output by pyHiM based on cells type (exocrine, beta, etc.). The analysis is
performed the following way :
1- based on DAPI images, perform registration of "rna" channel using "dna-HiM" channel as reference
2- for each ROI, load the rna images associated to the gene of interest and apply registration
3- using a pre-trained model in Ilastik, perform segmentation of the cells (for example beta or alpha).
4- filter and sort the traces based on the segmentation results and save them

Note :
- since registration & segmentation are performed in 3d, to save computation time and space, the images are reduced in
ALL DIRECTIONS by a factor equal to 2.
------------------------------------------------------------------------------------------------------------------------
"""

## FUNCTIONS ##

import os
import copy
import h5py
import numpy as np
import matplotlib
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import logging
import json
import shutil
from glob import glob
from tqdm import tqdm
from tifffile import imread, TiffWriter
from scipy import ndimage
from astropy.table import Table
from HiM_analysis.Traces.Filter_traces_class import FilterTraces
from HiM_analysis.Traces.Ilastik_mask import DefineMask

# matplotlib.use('TkAgg')
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Defines the configuration file where all the parameters are defined
# -------------------------------------------------------------------
configuration_path = '/home/fiche/Repositories/lab_scripts/HiM_analysis/Traces/Config_sorting.yml'
# configuration_path = '/home/jb/Repositories/lab_scripts/HiM_analysis/Traces/Config_sorting.yml'

# Indicate whether previous analysis should be reused
# ----------------------------------------------------
use_previous_registration_optimization = True
use_previous_registered_image = True
use_previous_segmentation = True


# Definition of the helper functions
# ----------------------------------

def check_folder_organization(folder_path):
    """ Reorganize the data according to the folder organization used for the RAMM microscope

    @param folder_path: (str) name of the folder to analyse
    """
    # check if the folder was already organized by ROI (like on the RAMM)
    folders = glob(os.path.join(folder_path, 'ROI*'))
    if len(folders) == 0:
        files = glob(os.path.join(folder_path, 'scan*_ROI.tif'))
        if files:
            for file in files:
                # for each file, retrieve from the filename the ROI and RT. Create an ROI folder and then an RT folder
                # where the data are moved
                path, filename = os.path.split(file)
                filename_digit = [int(s) for s in filename.split('_') if s.isdigit()]
                filename_string = [str(s) for s in filename.split('_') if not s.isdigit()]
                roi_number = filename_digit[1]
                rt_name = filename_string[1]
                roi_folder = os.path.join(path, f'ROI_{str(roi_number).zfill(3)}')
                if not os.path.exists(roi_folder):
                    os.mkdir(roi_folder)
                rt_folder = os.path.join(roi_folder, rt_name)
                os.mkdir(rt_folder)

                print(os.path.join(rt_folder, filename))
                shutil.move(file, os.path.join(rt_folder, filename))


def extract_roi_folder_info(folder_path):
    """ Analyse the path to the selected ROI folder and extract the associated digit number

    @param folder_path: path to the selected ROI folder
    @return: the name of the ROI folder as well the associated number
    """
    folder_name = os.path.basename(folder_path)
    folder_number = [int(s) for s in folder_name.split('_') if s.isdigit()]
    if len(folder_number) == 1:
        folder_number = folder_number[0]
    else:
        folder_number = None

    return folder_name, folder_number


def create_sub_folder(root_folder, sub_folder_name):
    """ Create a sub-folder in a destination folder

    @param root_folder: root folder where the sub-folder is created
    @param sub_folder_name: name of the sub-folder
    @return sub_folder_path: (str) name of the newly created sub_folder
    """
    sub_folder_path = os.path.join(root_folder, sub_folder_name)
    if not os.path.isdir(sub_folder_path):
        os.mkdir(sub_folder_path)

    return sub_folder_path


def check_file_exist(folder, file_name):
    """ Check if a file already exists in the indicated folder

    @param folder: folder where to look for the file
    @param file_name: name of the file we are looking for
    @return: boolean indicating whether file_name already exists
    """
    file = glob(os.path.join(folder, file_name))
    if file:
        return True
    else:
        return False


def load_pyhim_registration_shift(him_folder, roi_selected, file_name, channel):
    """ During pyHiM analysis, all the traces are registered with respect to a reference barcode, not the DAPI. The
    value for the registration of the DAPI channel calculated by pyHiM are then loaded here.

    @param him_folder: (str) path to the root folder where to find pyHiM analysis
    @param roi_selected: indicate the number associated to the selected ROI
    @param file_name: (str) name of the file created by pyHiM where the shift values have been saved
    @param channel: (str) name of channel (for us usually DAPI)
    @return: (np array) an array containing the values of the 2d shift allowing registration of DAPI images with
    respect to the reference barcode.
    """
    # define the path to the registration file
    path = os.path.join(him_folder, f'{roi_selected:03d}_ROI*', 'alignImages', file_name)
    file = glob(path)

    if len(file) == 1:
        with open(file[0]) as f:
            registration_dic = json.load(f)
    elif len(file) > 1:
        print('Too many registration files were found. The loading of the pyHiM registration dict is aborted.')
        return None
    else:
        print('No registration file was found. The loading of the pyHiM registration dict is aborted.')
        return None

    # from the dictionary, load the shift for the DAPI channel
    roi_key = f'ROI:{roi_selected:03d}'
    shift_dic = registration_dic[roi_key]
    if channel in shift_dic:
        shift = shift_dic[channel]
    else:
        shift = [0, 0]

    return shift


def sort_traces(trace_data, xy_shift, beta, alpha, pixel_dim, downsizing):
    """ Sort the traces based on the classes

    @param xy_shift: (np array) values of the registration shift from pyHiM (DAPI vs. reference channel)
    @param beta: (np array) mask of the beta cells
    @param alpha: (np array) mask of the alpha cells
    @param trace_data: (pd dataframe) contains the trace data
    @param downsizing: (int) indicate by which power of 2 the images are rescaled (to save space and time)
    @param pixel_dim: (list) indicate the pixel width (X,Y) and height (Z)
    @return: (pd dataframe) return the modified dataframe after labelling
    """
    # create a new column called 'label'
    trace_data['label'] = ''

    # defines the size of the alpha matrix
    lz, lx, ly = alpha.shape

    # check if the mask for the beta cells is 2d or 3d. If 2d convert the mask to 3d
    if len(beta.shape) == 2:
        beta = np.broadcast_to(beta, [lz, lx, ly])

    # load 3d localizations of each detected spot (in pixels). Rescale them according to the rescaling factor used
    # for DAPI registration and apply the registration shift introduced by pyHiM
    # scaling_factor = 2 ** self.downsizing_power * self.pixel_size[0]
    # traces_x = trace_data["x"] / scaling_factor
    # traces_y = trace_data["y"] / scaling_factor
    # traces_z = trace_data["z"] / self.pixel_size[1]

    scaling_factor = 2 ** downsizing
    traces_x = (trace_data["x"] / pixel_dim[0] - xy_shift[1]) / scaling_factor
    traces_y = (trace_data["y"] / pixel_dim[0] - xy_shift[0]) / scaling_factor
    traces_z = trace_data["z"] / pixel_dim[1]

    # for each trace, calculate its average 3d position and associate a specific label based on the maks calculated
    # for the alpha and beta cells
    unique_trace_id = trace_data["Trace_ID"].drop_duplicates()
    for trace_id in tqdm(unique_trace_id):

        index = trace_data.index[trace_data["Trace_ID"] == trace_id]
        x = np.mean(traces_x.loc[index])
        y = np.mean(traces_y.loc[index])
        z = np.mean(traces_z.loc[index])

        # check if the trace is within the border of the images for the beta segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            beta_label = beta[int(z), int(y), int(x)]
        else:
            beta_label = 0

        # check if the trace is within the border of the images for the alpha segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            alpha_label = alpha[int(z), int(y), int(x)]
        else:
            alpha_label = 0

        # assign  a label to the selected trace
        if alpha_label == 1:
            trace_label = 'alpha'
        elif (beta_label == 1) and (alpha_label != 1):
            trace_label = 'beta'
        elif (beta_label == 2) and (alpha_label != 1):
            trace_label = 'exocrine'
        else:
            trace_label = 'out'

        for idx in index:
            trace_data.at[idx, 'label'] = trace_label

    return trace_data


def sort_traces_alpha_beta_gamma(trace_data, xy_shift, beta, alpha, gamma, exocrine, pixel_dim, downsizing):
    """ Sort the traces based on the classes

    @param xy_shift: (np array) values of the registration shift from pyHiM (DAPI vs. reference channel)
    @param beta: (np array) mask of the beta cells
    @param alpha: (np array) mask of the alpha cells
    @param trace_data: (pd dataframe) contains the trace data
    @param downsizing: (int) indicate by which power of 2 the images are rescaled (to save space and time)
    @param pixel_dim: (list) indicate the pixel width (X,Y) and height (Z)
    @return: (pd dataframe) return the modified dataframe after labelling
    """
    # create a new column called 'label'
    trace_data['label'] = ''

    # defines the size of the alpha matrix
    lz, lx, ly = alpha.shape

    # defines the size of the gamma matrix
    # lz, lx, ly = gamma.shape

    # check if the mask for the beta cells is 2d or 3d. If 2d convert the mask to 3d
    if len(beta.shape) == 2:
        beta = np.broadcast_to(beta, [lz, lx, ly])

    # check if the mask for the beta cells is 2d or 3d. If 2d convert the mask to 3d
    if len(exocrine.shape) == 2:
        exocrine = np.broadcast_to(exocrine, [lz, lx, ly])

    # load 3d localizations of each detected spot (in pixels). Rescale them according to the rescaling factor used
    # for DAPI registration and apply the registration shift introduced by pyHiM
    # scaling_factor = 2 ** self.downsizing_power * self.pixel_size[0]
    # traces_x = trace_data["x"] / scaling_factor
    # traces_y = trace_data["y"] / scaling_factor
    # traces_z = trace_data["z"] / self.pixel_size[1]

    scaling_factor = 2 ** downsizing
    traces_x = (trace_data["x"] / pixel_dim[0] - xy_shift[1]) / scaling_factor
    traces_y = (trace_data["y"] / pixel_dim[0] - xy_shift[0]) / scaling_factor
    traces_z = trace_data["z"] / pixel_dim[1]

    # for each trace, calculate its average 3d position and associate a specific label based on the maks calculated
    # for the alpha and beta and gamma cells
    unique_trace_id = trace_data["Trace_ID"].drop_duplicates()
    for trace_id in tqdm(unique_trace_id):

        index = trace_data.index[trace_data["Trace_ID"] == trace_id]
        x = np.mean(traces_x.loc[index])
        y = np.mean(traces_y.loc[index])
        z = np.mean(traces_z.loc[index])

        # check if the trace is within the border of the images for the beta segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            beta_label = beta[int(z), int(y), int(x)]
        else:
            beta_label = 0

        # check if the trace is within the border of the images for the exocrine segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            exocrine_label = exocrine[int(z), int(y), int(x)]
        else:
            exocrine_label = 0

        # check if the trace is within the border of the images for the alpha segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            alpha_label = alpha[int(z), int(y), int(x)]
        else:
            alpha_label = 0

        # check if the trace is within the border of the images for the gamma segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly) and (z > 0) and (z < lz):
            gamma_label = gamma[int(z), int(y), int(x)]
        else:
            gamma_label = 0

        # assign  a label to the selected trace
        if (alpha_label == 1) and (gamma_label != 1) and (exocrine_label != 1):
            trace_label = 'alpha'
            # print('alpha found')
        elif (beta_label == 2) and (alpha_label != 1) and (gamma_label != 1) and (exocrine_label != 1):
            trace_label = 'beta'
            # print('beta found')
        elif (alpha_label != 1) and (gamma_label == 1) and (exocrine_label != 1):
            trace_label = 'gamma'
            # print('gamma found')
        elif (beta_label == 1) and (alpha_label != 1) and (gamma_label != 1) and (exocrine_label == 1):
            trace_label = 'exocrine'
        elif (beta_label == 0):
            trace_label = 'out'
        else:
            trace_label = 'out'

        for idx in index:
            trace_data.at[idx, 'label'] = trace_label

    return trace_data


def sort_traces_endocrine_exocrine(trace_data, xy_shift, exocrine, pixel_dim, downsizing):
    """ Sort the traces based on the classes

    @param xy_shift: (np array) values of the registration shift from pyHiM (DAPI vs. reference channel)
    @param trace_data: (pd dataframe) contains the trace data
    @param downsizing: (int) indicate by which power of 2 the images are rescaled (to save space and time)
    @param pixel_dim: (list) indicate the pixel width (X,Y) and height (Z)
    @return: (pd dataframe) return the modified dataframe after labelling
    """
    # create a new column called 'label'
    trace_data['label'] = ''

    # defines the size of the alpha matrix
    lx, ly = exocrine.shape

    scaling_factor = 2 ** downsizing
    traces_x = (trace_data["x"] / pixel_dim[0] - xy_shift[1]) / scaling_factor
    traces_y = (trace_data["y"] / pixel_dim[0] - xy_shift[0]) / scaling_factor
    # traces_z = trace_data["z"] / pixel_dim[1]

    # # for each trace, calculate its average 3d position and associate a specific label based on the maks calculated
    # # for the alpha and beta and gamma cells
    unique_trace_id = trace_data["Trace_ID"].drop_duplicates()
    for trace_id in tqdm(unique_trace_id):

        index = trace_data.index[trace_data["Trace_ID"] == trace_id]
        x = np.mean(traces_x.loc[index])
        y = np.mean(traces_y.loc[index])

        # check if the trace is within the border of the images for the exocrine segmentation
        if (x > 0) and (x < lx) and (y > 0) and (y < ly):
            exocrine_label = exocrine[int(y), int(x)]
        else:
            exocrine_label = 0

        # assign  a label to the selected trace
        if (exocrine_label != 1):
            trace_label = 'endorcine'
        else:
            trace_label = 'exocrine'
        for idx in index:
            trace_data.at[idx, 'label'] = trace_label
    return trace_data


def plot_labeled_traces(root_folder, trace_data, labels, xy_shift, im_name, pixel_dim, tag=None):
    """ overlay the traces with an input image im. Each trace is color coded according to its associated label.

    @param root_folder: (str) root folder where to save the figure
    @param trace_data: (pd dataframe) input trace data
    @param labels: (dict) indicate for each label the associated pixel value
    @param xy_shift: (np array) values of the registration shift from pyHiM (DAPI vs. reference channel)
    @param im_name: (str) name to the image file selected as background
    @param pixel_dim: (list) indicate the pixel width (X,Y) and height (Z)
    @param tag: (str) add a tag to the output figure name
    """
    # rescale the positions associated to each trace
    traces_x = (trace_data["x"] / pixel_dim[0] - xy_shift[1])
    traces_y = (trace_data["y"] / pixel_dim[0] - xy_shift[0])

    # load the registered raw image and format it
    im_path = os.path.join(root_folder, im_name)
    bkgd_im = imread(im_path)

    # if the image selected as background is 3d, perform a mip
    if len(bkgd_im.shape) != 2:
        bkgd_im = np.max(bkgd_im, axis=0)

    # overlay the traces with the image
    colors = ['white', 'red', 'blue', 'black', 'yellow', 'green', 'orange']
    plt.close('all')
    plt.imshow(bkgd_im)
    for n, trace_label in enumerate(labels):
        plt.scatter(traces_x[trace_data['label'] == trace_label], traces_y[trace_data['label'] == trace_label],
                    s=2, c=colors[n])

    # save the figure
    if tag is None:
        figure_name = 'Trace_detection.png'
    else:
        figure_name = f'Trace_detection_{tag}.png'
    plt.savefig(os.path.join(root_folder, figure_name))


def mask_segmentation(root_folder, param, _mask_ref):
    """ helper function used to calculate the mask (for filtering the traces)

    @param root_folder: (str) folder where the data will be saved
    @param param: (dict) parameters for the selected type of cells
    @param _mask_ref: (class reference) reference to the class handling mask segmentation and registration
    @return: (np array) stack of raw images used for mask segmentation
    """
    h5_stack_available = check_file_exist(root_folder, param["h5_filename"])
    segmentation_file_name = f'{os.path.splitext(param["h5_filename"])[0]}-{param["h5_groupname"]}-raw_formatted_' \
                             f'{param["ilastik_output_format"]}.tiff'
    segmentation_available = check_file_exist(root_folder, segmentation_file_name)

    # load the stack and save it as a h5 file
    if not h5_stack_available:
        rna_file = glob(os.path.join(rna_data_folder,
                                     roi_name,
                                     param['rna_data_folder'],
                                     f"*{param['rna_data_folder']}*.tif"))

        stack = _mask_ref.load_stack(rna_file, rna_total_number_channels, param['rna_channel_to_keep'])
        _mask_ref.save_h5_image(stack, param["h5_filename"], f'{param["h5_groupname"]}/raw')

    else:
        stack = _mask_ref.open_h5_image(root_folder, param["h5_filename"], f'{param["h5_groupname"]}/raw')

    # performed segmentation before registration (to avoid artefacts created by the large patch of pixels
    # with value set to zero)
    if (not segmentation_available) or (not use_previous_segmentation):
        if param['mode'] == "manual":
            _, mask = _mask_ref.load_manual_2d_mask(param['mask_path'], roi_name, [1, 2])
            _mask_ref.save_tiff_image(mask, segmentation_file_name, rescale=False)
            # _mask_ref.load_beta_segmentation(roi_name, segmentation_file_name)
        else:
            # _mask_ref.ilastik_beta_segmentation(stack)
            _mask_ref.ilastik_segmentation(stack,
                                           param['apply_MIP'],
                                           param['ilastik_model'],
                                           param['ilastik_output_format'],
                                           param["h5_filename"],
                                           f'{param["h5_groupname"]}',
                                           "raw_formatted")

    return stack


def load_islet_segmentation(roi_folder, islet_classification_model, results_folder, _mask_ref):
    path_mask = os.path.join(islet_classification_model, roi_folder, "*mask.tif")
    # print(path_mask)
    mask = glob(path_mask)
    # print(mask)
    if mask:
        im_mask = imread(mask)
        im_mask[im_mask == 0] = 1
        im_mask[im_mask != 1] = 2
        # keys = get_h5_keys(results_folder, "islet_stacks.h5", group="islet_rna")
        # self.save_tiff_image(im_mask, os.path.join(self.saving_path, mask_name))
        im_mask[im_mask == 2] = 0
        im_mask[im_mask > 0] = 1
        im_mask = ndimage.grey_opening(im_mask, size=(3, 3)).astype(int)
        im_mask = ndimage.binary_fill_holes(im_mask).astype(int)
        # save the masks
        _mask_ref.save_tiff_image(im_mask, "Islet_mask_manual_segmentation.tiff", rescale=False)
    else:
        im_mask = np.zeros((2048, 2048))
        _mask_ref.save_tiff_image(im_mask, "Islet_mask_manual_segmentation.tiff", rescale=False)
    return im_mask


def mask_registration(root_folder, stack, param, _mask_ref):
    """ Helper function used to apply registration on the raw image (2d of 3d) and the segmented mask

    @param stack: (np array) stack of raw images used for mask segmentation
    @param root_folder: (str) folder where the data will be saved
    @param param: (dict) parameters for the selected type of cells
    @param _mask_ref: (class reference) reference to the class handling mask segmentation and registration
    """
    registration_available = check_file_exist(root_folder,
                                              f'{param["root_output_name"]}_segmented_registered.tiff')
    segmentation_file_name = f'{os.path.splitext(param["h5_filename"])[0]}-{param["h5_groupname"]}-raw_formatted_' \
                             f'{param["ilastik_output_format"]}.tiff'

    # perform 3d / 2d registration of the raw image and the segmented images. Save them as tiff
    if (not registration_available) or (not use_previous_registered_image):
        stack_registered = _mask_ref.apply_3d_registration(stack)
        _mask_ref.save_tiff_image(stack_registered, f'{param["root_output_name"]}_registered.tiff', rescale=True)

        im_segmented = imread(os.path.join(root_folder, segmentation_file_name))
        if param['apply_MIP']:
            im_segmented_registered = _mask_ref.apply_2d_registration(im_segmented, downsize=True)
        else:
            im_segmented = _mask_ref.duplicate_frame_stack(im_segmented, stack.shape[0])
            im_segmented_registered = _mask_ref.apply_3d_registration(im_segmented, downsize=True)

        im_segmented_registered = 100 * im_segmented_registered
        im_segmented_registered[im_segmented_registered < 0] = 0
        _mask_ref.save_tiff_image(im_segmented_registered,
                                  f'{param["root_output_name"]}_segmented_registered.tiff',
                                  rescale=False)


def create_alpha_beta_gamma_masks(root_folder, alpha_name, beta_name, gamma_name, gauss_sigma=5,
                                  alpha_probability_threshold=30,
                                  alpha_size_threshold=50, gamma_probability_threshold=30, gamma_size_threshold=50):
    """ Analyze the segmentation for the alpha and beta cells in order to create two masks for filtering the traces.
    @param root_folder: (str) indicate the folder where the data will be saved
    @param alpha_name: (str) name of the registered and segmented stack for the alpha cells
    @param beta_name: (str) name of the registered and segmented images for the beta cells
    @param gauss_sigma: (int) size in pixel of the kernel for the gaussian filter
    @param alpha_probability_threshold: (int) minimum value for the probability of the alpha cells
    @param alpha_size_threshold: (int) minimum size (in pixel) of the objects detected on the alpha cells
    """
    beta_segmentation_path = os.path.join(root_folder, beta_name)
    alpha_segmentation_path = os.path.join(root_folder, alpha_name)
    gamma_segmentation_path = os.path.join(root_folder, gamma_name)
    beta_segmentation = imread(beta_segmentation_path)
    alpha_segmentation = imread(alpha_segmentation_path)
    gamma_segmentation = imread(gamma_segmentation_path)

    # # create a mask for the ilet (if it exists) by performing a MIP of the segmentation and filling up the holes
    # ilet_mask = np.copy(beta_segmentation)
    # ilet_mask[ilet_mask == 200] = 0
    # ilet_mask[ilet_mask > 0] = 1
    # ilet_mask = np.max(ilet_mask, axis=0)
    # ilet_mask = ndimage.grey_opening(ilet_mask, size=(3, 3)).astype(int)
    # ilet_mask = ndimage.binary_fill_holes(ilet_mask).astype(int)

    # create the mask for the beta cells
    beta_cell_mask = np.copy(beta_segmentation)

    beta_cell_mask[beta_cell_mask == 200] = 0
    beta_cell_mask[beta_cell_mask > 0] = 1
    if len(beta_cell_mask.shape) == 3:
        beta_cell_mask = ndimage.grey_opening(beta_cell_mask, size=(3, 3, 3)).astype(int)
    else:
        beta_cell_mask = ndimage.grey_opening(beta_cell_mask, size=(3, 3)).astype(int)
    beta_cell_mask = beta_cell_mask + 1
    beta_cell_mask[beta_segmentation == 0] = 0

    # create the mask for the alpha cells
    alpha_cell_mask = np.zeros(alpha_segmentation.shape)
    for frame in range(alpha_segmentation.shape[0]):
        im = alpha_segmentation[frame, :, :]

        # apply a gaussian filter to smooth the image
        im = ndimage.gaussian_filter(im, gauss_sigma)

        # alpha cells are in the ilet, therefore they should be located within the mask defined for the beta cells.
        # A minimum probability "alpha_probability_threshold" is also used as threshold.
        im[im < alpha_probability_threshold] = 0
        im[im > 0] = 1
        im = ndimage.binary_fill_holes(im).astype(int)

        # for each frame, the objects are localized and filtered according to their size (in number of pixels)
        labeled_im, nr_objects = ndimage.label(im)
        unique_label_values = np.unique(labeled_im)
        for label_value in unique_label_values:
            objects_size = np.sum(labeled_im == label_value)
            if objects_size < alpha_size_threshold:
                im[labeled_im == label_value] = 0

        alpha_cell_mask[frame, :, :] = im

    # create the mask for the gamma cells
    gamma_cell_mask = np.zeros(gamma_segmentation.shape)
    for frame in range(gamma_segmentation.shape[0]):
        im = gamma_segmentation[frame, :, :]

        # apply a gaussian filter to smooth the image
        im = ndimage.gaussian_filter(im, gauss_sigma)

        # gamma cells are in the ilet, therefore they should be located within the mask defined for the beta cells.
        # A minimum probability "gamma_probability_threshold" is also used as threshold.
        im[im < gamma_probability_threshold] = 0
        im[im > 0] = 1
        im = ndimage.binary_fill_holes(im).astype(int)

        # for each frame, the objects are localized and filtered according to their size (in number of pixels)
        labeled_im, nr_objects = ndimage.label(im)
        unique_label_values = np.unique(labeled_im)
        for label_value in unique_label_values:
            objects_size = np.sum(labeled_im == label_value)
            if objects_size < gamma_size_threshold:
                im[labeled_im == label_value] = 0

        gamma_cell_mask[frame, :, :] = im

    # save the masks
    im_path = os.path.join(root_folder, "beta_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(beta_cell_mask.astype(np.uint16))

    im_path = os.path.join(root_folder, "alpha_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(alpha_cell_mask.astype(np.uint16))

    im_path = os.path.join(root_folder, "gamma_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(gamma_cell_mask.astype(np.uint16))

    return alpha_cell_mask, beta_cell_mask, gamma_cell_mask


def create_alpha_beta_masks(root_folder, alpha_name, beta_name, gauss_sigma=5, alpha_probability_threshold=30,
                            alpha_size_threshold=50):
    """ Analyze the segmentation for the alpha and beta cells in order to create two masks for filtering the traces.
    @param root_folder: (str) indicate the folder where the data will be saved
    @param alpha_name: (str) name of the registered and segmented stack for the alpha cells
    @param beta_name: (str) name of the registered and segmented images for the beta cells
    @param gauss_sigma: (int) size in pixel of the kernel for the gaussian filter
    @param alpha_probability_threshold: (int) minimum value for the probability of the alpha cells
    @param alpha_size_threshold: (int) minimum size (in pixel) of the objects detected on the alpha cells
    """
    beta_segmentation_path = os.path.join(root_folder, beta_name)
    alpha_segmentation_path = os.path.join(root_folder, alpha_name)
    beta_segmentation = imread(beta_segmentation_path)
    alpha_segmentation = imread(alpha_segmentation_path)

    # create the mask for the beta cells
    beta_cell_mask = np.copy(beta_segmentation)
    beta_cell_mask[beta_cell_mask == 200] = 0
    beta_cell_mask[beta_cell_mask > 0] = 1
    if len(beta_cell_mask.shape) == 3:
        beta_cell_mask = ndimage.grey_opening(beta_cell_mask, size=(3, 3, 3)).astype(int)
    else:
        beta_cell_mask = ndimage.grey_opening(beta_cell_mask, size=(3, 3)).astype(int)
    beta_cell_mask = beta_cell_mask + 1
    beta_cell_mask[beta_segmentation == 0] = 0

    # create the mask for the alpha cells
    alpha_cell_mask = np.zeros(alpha_segmentation.shape)
    for frame in range(alpha_segmentation.shape[0]):
        im = alpha_segmentation[frame, :, :]

        # apply a gaussian filter to smooth the image
        im = ndimage.gaussian_filter(im, gauss_sigma)

        # alpha cells are in the ilet, therefore they should be located within the mask defined for the beta cells.
        # A minimum probability "alpha_probability_threshold" is also used as threshold.
        im[im < alpha_probability_threshold] = 0
        im[im > 0] = 1
        im = ndimage.binary_fill_holes(im).astype(int)

        # for each frame, the objects are localized and filtered according to their size (in number of pixels)
        labeled_im, nr_objects = ndimage.label(im)
        unique_label_values = np.unique(labeled_im)
        for label_value in unique_label_values:
            objects_size = np.sum(labeled_im == label_value)
            if objects_size < alpha_size_threshold:
                im[labeled_im == label_value] = 0

        alpha_cell_mask[frame, :, :] = im

    # save the masks
    im_path = os.path.join(root_folder, "beta_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(beta_cell_mask.astype(np.uint16))

    im_path = os.path.join(root_folder, "alpha_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(alpha_cell_mask.astype(np.uint16))

    return alpha_cell_mask, beta_cell_mask

    # save the masks
    im_path = os.path.join(root_folder, "beta_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(beta_cell_mask.astype(np.uint16))

    im_path = os.path.join(root_folder, "alpha_mask.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(alpha_cell_mask.astype(np.uint16))

    return alpha_cell_mask, beta_cell_mask


def mask_removing_false_positive_signal(root_folder, islet_mask, alpha_mask, beta_mask, gamma_mask):
    """ Analyze the segmentation for the alpha and beta cells and gamma and check if the signal is within islet
    """

    # Bin the islet mask
    resize_shape = (int(islet_mask.shape[1] / 2 ** 1), int(islet_mask.shape[1] / 2 ** 1))
    shape = (resize_shape[0], islet_mask.shape[0] // resize_shape[0],
             resize_shape[1], islet_mask.shape[1] // resize_shape[1])
    islet_mask_binned = islet_mask.reshape(shape).mean(-1).mean(1)

    # =============================ATTENTION===================================== #
    # (for Olivier) in case RNA & DNA images were flipped ONLY in first experiment
    # islet_mask_binned_flipped = np.flipud(islet_mask_binned)
    # islet_mask_binned_flipped = np.fliplr(islet_mask_binned_flipped)
    # Exocrine Mask
    # exocrine_mask = np.logical_not(islet_mask_binned_flipped).astype(int)
    # =============================ATTENTION===================================== #

    exocrine_mask = np.logical_not(islet_mask_binned).astype(int)

    # Mask Beta corrected in 2D
    indices = (islet_mask_binned == 0) & (beta_mask == 2)
    beta_mask[indices] = 1
    beta_cell_mask_corrected = beta_mask

    # save the masks
    im_path = os.path.join(root_folder, "beta_mask_corrected.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(beta_cell_mask_corrected.astype(np.uint16))

    # Mask Alpha corrected in 3D
    for i in range(alpha_mask.shape[0]):
        indices = (islet_mask_binned == 0) & (alpha_mask[i, :, :] == 1)
        alpha_mask[i, :, :][indices] = 0
    alpha_cell_mask_corrected = alpha_mask

    # save the masks
    im_path = os.path.join(root_folder, "alpha_mask_corrected.tiff")
    with TiffWriter(im_path) as tif:
        tif.write(alpha_cell_mask_corrected.astype(np.uint16))

    # Mask Gamma corrected in 3D
    for i in range(gamma_mask.shape[0]):
        indices = (islet_mask_binned == 0) & (gamma_mask[i, :, :] == 1)
        gamma_mask[i, :, :][indices] = 0
    gamma_cell_mask_corrected = gamma_mask

    # save the masks
    im_path = os.path.join(root_folder, "gamma_mask_corrected.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(gamma_cell_mask_corrected.astype(np.uint16))

    # save the masks
    im_path = os.path.join(root_folder, "exocrine_mask_corrected.tiff")
    with TiffWriter(im_path) as tif:
        tif.save(exocrine_mask.astype(np.uint16))

    return alpha_cell_mask_corrected, beta_cell_mask_corrected, gamma_cell_mask_corrected, exocrine_mask


# Load parameters
# ---------------
with open(configuration_path, "r") as configuration_file:
    parameters = yaml.load(configuration_file, Loader=yaml.FullLoader)

rna_data_folder = parameters['rna_data_folder']
rna_ROI_folder_template = parameters['rna_ROI_folder_template']
rna_total_number_channels = parameters['rna_total_number_channels']

binning = parameters['binning']
rescaling_Z = parameters['rescaling_Z']

beta = parameters['beta']
alpha = parameters['alpha']
gamma = parameters['gamma']

islet_segmentation_mode = parameters['islet_segmentation_mode']
islet_ROI_folder_template = parameters['islet_ROI_folder_template']
islet_classification_model = parameters['islet_classification_model']

dest_folder = parameters['dest_folder']

reference_DAPI_folder = parameters['reference_DAPI_folder']
template_reference_DAPI_file = parameters['template_reference_DAPI_file']
number_channel_reference_stack = parameters['number_channel_reference_stack']
reference_channel = parameters['reference_channel']
target_DAPI_folder = parameters['target_DAPI_folder']
template_target_DAPI_file = parameters['template_target_DAPI_file']
number_channel_target_stack = parameters['number_channel_target_stack']
target_channel = parameters['target_channel']

him_traces_folder = parameters['him_traces_folder']
RT_reference_registration_dic = parameters['pyHiM_registration_ref_fiducial']
trace_ROI_pattern = parameters['trace_ROI_pattern']
trace_file_pattern = parameters['trace_file_pattern']
overlapping_threshold = parameters['overlapping_threshold']
# minimum_efficiency = parameters['minimum_efficiency']
pixel_size = parameters['pixel_size']
downsizing_power = parameters['registration_parameters']['downsizing_power']
trace_labels = parameters['trace_labels']

# Check the structure of the selected folders. If the data are not organized by ROI/RT folders, reorganize the data
# -----------------------------------------------------------------------------------------------------------------
check_folder_organization(reference_DAPI_folder)
check_folder_organization(target_DAPI_folder)
check_folder_organization(rna_data_folder)

# Look for all the folders corresponding to an ROI in the rna_folder and perform the analysis in each one
# -------------------------------------------------------------------------------------------------------
concatenated_traces = []
concatenated_traces_endo_exo = []

# RNA signal segmentation
roi_folders = sorted(glob(os.path.join(rna_data_folder, rna_ROI_folder_template)))

# # Islet segmentation
# roi_folders = sorted(glob(os.path.join(islet_classification_model, islet_ROI_folder_template)))

if not roi_folders:
    logger.warning('No folders were found for the data. The analysis is aborted.')
    exit()

for roi in roi_folders:
    #
    # # Uncomment the following conditions to run the analysis on a specific ROI
    if os.path.basename(roi) == "ROI_002" or os.path.basename(roi) == "ROI_013":
        print(roi)
        continue

    # create in the destination folder a sub-folder where the results will be saved for the selected roi
    roi_name, roi_number = extract_roi_folder_info(roi)
    results_folder = create_sub_folder(dest_folder, roi_name)
    # save the parameters file
    saved_config_path = os.path.join(results_folder, 'analysis_parameters.yml')
    with open(saved_config_path, 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False)

    # instantiate the DefineMask class and define the saving path
    _mask = DefineMask(configuration_path)
    _mask.saving_path = results_folder

    # perform optimization of the registration parameters on DAPI data (either load a previous analysis or run the
    # registration pipeline)
    # ---------------------
    print(f'STEP 1 : Image registration for {roi_name}')
    print('-------------------------------------------')

    registration_available = check_file_exist(results_folder, "registration_parameters.json")

    if registration_available and use_previous_registration_optimization:
        _mask.load_registration_dic(results_folder)
        print(f'Optimized registration parameters for {roi_name} loaded')
    else:
        # define the path to the reference and target stacks. Load the images and optimize registration parameters.
        # Save the parameters in a json file.
        reference_path = glob(os.path.join(reference_DAPI_folder, roi_name, '**', template_reference_DAPI_file),
                              recursive=True)
        target_path = glob(os.path.join(target_DAPI_folder, roi_name, '**', template_target_DAPI_file),
                           recursive=True)

        im_reference = _mask.load_stack(reference_path, number_channel_reference_stack, reference_channel)
        # # (for Olivier) in case RNA & DNA images were flipped
        # for frame in range(im_reference.shape[0]):
        #	 im_reference[frame, :, :] = np.flipud(im_reference[frame, :, :])
        #	 im_reference[frame, :, :] = np.fliplr(im_reference[frame, :, :])
        im_to_align = _mask.load_stack(target_path, number_channel_target_stack, target_channel)

        _mask.register_dapi_images(im_reference, im_to_align)
        print(f'Registration parameters optimized for {roi_name}')

    # perform segmentation of beta cells and calculate the mask
    # ---------------------------------------------------------
    print(f'STEP 2 : Performing beta cells segmentation for {roi_name}')
    print('-----------------------------------------------------------')

    beta_stack = mask_segmentation(results_folder, beta, _mask)
    mask_registration(results_folder, beta_stack, beta, _mask)

    # perform segmentation of alpha cells
    # -----------------------------------
    print(f'STEP 3 : Performing alpha cells segmentation for {roi_name}')
    print('------------------------------------------------------------')

    alpha_stack = mask_segmentation(results_folder, alpha, _mask)
    mask_registration(results_folder, alpha_stack, alpha, _mask)

    # perform segmentation of alpha cells
    # -----------------------------------
    print(f'STEP 4 : Performing gamma cells segmentation for {roi_name}')
    print('------------------------------------------------------------')

    gamma_stack = mask_segmentation(results_folder, gamma, _mask)
    mask_registration(results_folder, gamma_stack, gamma, _mask)

    print(f'STEP 5 : Performing islet segmentation for {roi_name}')
    print('-----------------------------------------------------------')

    roi_folders = sorted(glob(os.path.join(islet_classification_model, islet_ROI_folder_template)))

    h5_stack_available = check_file_exist(results_folder, "Islet.h5")
    islet_segmentation_file_name = "Islet.tiff"

    if islet_segmentation_mode == "manual":
        islet_mask = load_islet_segmentation(roi_name, islet_classification_model, results_folder, _mask)
    # print(segmented_mask_islet)

    print(f'STEP 6 : Masking Alpha Beta and Gamma cells for {roi_name}')
    print('------------------------------------------------------------')

    # Analyze the beta / alpha / gamma cells and compute the masks
    # ----------------------------------------------------
    alpha_mask, beta_mask, gamma_mask = create_alpha_beta_gamma_masks(results_folder,
                                                                      "alpha_segmented_registered.tiff",
                                                                      "beta_segmented_registered.tiff",
                                                                      "gamma_segmented_registered.tiff",
                                                                      gauss_sigma=5,
                                                                      alpha_probability_threshold=30,
                                                                      alpha_size_threshold=50,
                                                                      gamma_probability_threshold=30,
                                                                      gamma_size_threshold=50)

    print(f'STEP 7 : Removing false positive and false negative for {roi_name}')
    # Islet_mask is used to filtered alpha beta and gamma cells that fall outside islet mask
    print('---------------------------------------------------------')

    alpha_mask_corrected, beta_mask_corrected, gamma_mask_corrected, exocrine_mask = mask_removing_false_positive_signal(
        results_folder,
        islet_mask,
        alpha_mask,
        beta_mask,
        gamma_mask)

    # Open the trace file associated to the ROI. Since the traces have been aligned with respect to an RT of
    # reference, the values need to be shifted with respect to the DAPI images. Once the traces have been
    # recalculated, they are finally sorted based on the segmentation of beta and alpha cells.
    # ---------------------------------------------------------------------------------------
    print(f'STEP 8 : Analyzing and sorting the traces for {roi_name}')
    print('---------------------------------------------------------')

    # instantiate the class for loading the traces
    trace_path = os.path.join(him_traces_folder, f'{roi_number:03d}_ROI*', 'buildsPWDmatrix')
    trace_filename = f'{trace_file_pattern}{roi_number}.ecsv'
    print(trace_filename)
    _trace = FilterTraces(trace_path, trace_filename, results_folder)
    trace_endo_exo = FilterTraces(trace_path, trace_filename, results_folder)

    # load the file output by pyhim for the registration of DAPI vs. RT
    RT_ref_shift = load_pyhim_registration_shift(him_traces_folder, roi_number, RT_reference_registration_dic, 'DAPI')

    # if a trace file was found, perform sorting
    if _trace.data is not None:
        n_traces_total = _trace.n_traces_total

        # perform the sorting based on the raw trace data (no filtering)
        labeled_traces_corrected = sort_traces_alpha_beta_gamma(_trace.data, RT_ref_shift, beta_mask_corrected,
                                                                alpha_mask_corrected, gamma_mask_corrected,
                                                                exocrine_mask, pixel_size, downsizing_power)
        labeled_traces_endo_exo = sort_traces_endocrine_exocrine(trace_endo_exo.data, RT_ref_shift, exocrine_mask,
                                                                 pixel_size, downsizing_power)

        concatenated_traces.append(labeled_traces_corrected)  # append all traces into labeled_traces
        concatenated_traces_endo_exo.append(labeled_traces_endo_exo)  # append all traces into labeled_traces

        _trace.save_to_astropy(labeled_traces_corrected, tag='raw')
        trace_endo_exo.save_to_astropy(labeled_traces_endo_exo, tag='raw_endo_exo')

        # Plot Cell identify
        plot_labeled_traces(results_folder, labeled_traces_corrected, trace_labels,
                            RT_ref_shift, "alpha_registered.tiff",
                            pixel_size, tag="raw")

print('STEP 9 : Analyzing and filter all concatenated traces')
print('-----------------------------------------------------')

# concatenate all the labeled traces into a single dataframe and save them into a specific astropy files
concatenated_traces = pd.concat(concatenated_traces)
trace_data = Table.from_pandas(concatenated_traces)
trace_data.write(os.path.join(dest_folder, f'Trace_all_ROIs.ecsv'), overwrite=True)

# load the traces, compute the statistics and save individual trace file based on labels
_trace = FilterTraces(dest_folder, f'Trace_all_ROIs.ecsv', dest_folder, threshold=overlapping_threshold)
_trace.trace_statistics(save=True, tag="raw")
_trace.save_individual_labels(_trace.data)

# perform a "hard" filtering where all the traces with at least one duplicated barcode are discarded. Then,
# based on the masks calculated before, sort the traces and assign a specific label to each.
print('performing hard filtering ...')
hard_filtered_traces = _trace.hard_filtering()
_trace.save_to_astropy(hard_filtered_traces, tag="hard_filtering")
_trace.save_individual_labels(hard_filtered_traces, tag="hard_filtering")

# select the traces with no duplicates (and at least 2 barcodes) and calculate the distance threshold used
# later to filter the traces.
trace_wo_duplicates = _trace.select_traces_wo_duplicates(_trace.data)
pwd_min, pwd_max = _trace.calculate_pwd_threshold(trace_wo_duplicates, verbose=True, save=True)
print(
    f'Initial analysis returned {len(trace_wo_duplicates)}/{_trace.n_traces_total} traces with no duplicates & at '
    f'least 2 barcodes')

# based on the threshold defines above, analyze each trace in order to remove unspecific detections &
# separate traces that have been clustered together
filtered_traces = _trace.filter_traces(verbose=False)
trace_wo_duplicates = _trace.select_traces_wo_duplicates(filtered_traces)
print(
    f'After filtering, {len(trace_wo_duplicates)} traces are found with no duplicates '
    f'& at least 2 barcodes')

# filter the traces by removing duplicated barcodes and checking the number of barcodes is above the
# threshold mentioned in the parameters
filtered_traces, n_overlapping_bc = _trace.detect_overlapping_barcodes(filtered_traces, verbose=True,
                                                                       save=True)
trace_wo_duplicates = _trace.select_traces_wo_duplicates(filtered_traces)
print(
    f'A total of {n_overlapping_bc} overlapping barcodes were found. In total, {len(trace_wo_duplicates)} '
    f'traces are now found without duplicates and at least 2 barcodes')

final_traces, stat = _trace.remove_duplicates(filtered_traces)
print(
    f'After completion of the procedure, {stat[0]} traces were discarded, {stat[1]} traces were kept after '
    f'removing the duplicated barcodes and {stat[2]} were found without any duplicates.')

_trace.save_to_astropy(final_traces, tag="filtered")
_trace.save_individual_labels(final_traces, tag="filtered")

# Same but for endo / Exo

# concatenate all the labeled traces into a single dataframe and save them into a specific astropy files
concatenated_traces_endo_exo = pd.concat(concatenated_traces_endo_exo)
trace_data_endo_exo = Table.from_pandas(concatenated_traces_endo_exo)
trace_data_endo_exo.write(os.path.join(dest_folder, f'Trace_all_ROIs_endo_exo.ecsv'), overwrite=True)

# load the traces, compute the statistics and save individual trace file based on labels
trace_data_endo_exo = FilterTraces(dest_folder, f'Trace_all_ROIs_endo_exo.ecsv', dest_folder,
                                   threshold=overlapping_threshold)
trace_data_endo_exo.trace_statistics(save=True, tag="raw")
trace_data_endo_exo.save_individual_labels(trace_data_endo_exo.data)

# perform a "hard" filtering where all the traces with at least one duplicated barcode are discarded. Then,
# based on the masks calculated before, sort the traces and assign a specific label to each.
print('performing hard filtering ...')
hard_filtered_traces_endo_exo = trace_data_endo_exo.hard_filtering()
trace_data_endo_exo.save_to_astropy(hard_filtered_traces_endo_exo, tag="hard_filtering")
trace_data_endo_exo.save_individual_labels(hard_filtered_traces_endo_exo, tag="hard_filtering")

# select the traces with no duplicates (and at least 2 barcodes) and calculate the distance threshold used
# later to filter the traces.
trace_wo_duplicates_endo_exo = trace_data_endo_exo.select_traces_wo_duplicates(trace_data_endo_exo.data)
pwd_min, pwd_max = trace_data_endo_exo.calculate_pwd_threshold(trace_wo_duplicates_endo_exo, verbose=True, save=True)
print(
    f'Initial analysis returned {len(trace_wo_duplicates_endo_exo)}/{trace_data_endo_exo.n_traces_total} traces with no duplicates & at '
    f'least 2 barcodes')

# based on the threshold defines above, analyze each trace in order to remove unspecific detections &
# separate traces that have been clustered together
filtered_traces_endo_exo = trace_data_endo_exo.filter_traces(verbose=False)
trace_wo_duplicates_endo_exo = trace_data_endo_exo.select_traces_wo_duplicates(filtered_traces_endo_exo)
print(
    f'After filtering, {len(trace_wo_duplicates_endo_exo)} traces are found with no duplicates '
    f'& at least 2 barcodes')

# filter the traces by removing duplicated barcodes and checking the number of barcodes is above the
# threshold mentioned in the parameters
filtered_traces_endo_exo, n_overlapping_bc = trace_data_endo_exo.detect_overlapping_barcodes(filtered_traces_endo_exo,
                                                                                             verbose=True,
                                                                                             save=True)
trace_wo_duplicates_endo_exo = trace_data_endo_exo.select_traces_wo_duplicates(filtered_traces_endo_exo)
print(
    f'A total of {n_overlapping_bc} overlapping barcodes were found. In total, {len(trace_wo_duplicates_endo_exo)} '
    f'traces are now found without duplicates and at least 2 barcodes')

final_traces_endo_exo, stat = trace_data_endo_exo.remove_duplicates(filtered_traces_endo_exo)
print(
    f'After completion of the procedure, {stat[0]} traces were discarded, {stat[1]} traces were kept after '
    f'removing the duplicated barcodes and {stat[2]} were found without any duplicates.')

trace_data_endo_exo.save_to_astropy(final_traces_endo_exo, tag="filtered")
trace_data_endo_exo.save_individual_labels(final_traces_endo_exo, tag="filtered")

print('-----------------------------------------------------')
print('------------------------DONE-------------------------')
print('-----------------------------------------------------')
