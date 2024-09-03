# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Mon August 1, 2022
------------------------------------------------------------------------------------------------------------------------
Run locally or on lopevi using labscript conda environment.

This code is used to sort the traces output by pyHiM based on cells type (exocrine, beta, etc.). The analysis is
performed the following way :
1- based on DAPI images, perform registration of "rna" channel using "dna-HiM" channel as reference
2- for each ROI, load the rna images associated to ins gene expression (RT7 or RT24) and apply registration
3- using a pre-trained model in Ilastik, perform segmentation of beta cells (alpha cell segmentation is also possible
but 3d registration is required for sorting the cells, and it is not available yet).
4- sort the traces based on the segmentation results and save them
------------------------------------------------------------------------------------------------------------------------
"""

import os
import h5py
import numpy as np
import yaml
import logging
from functools import wraps
from time import time
from glob import glob
from tqdm import tqdm
from tifffile import imread
from HiM_analysis.image_registration.Image_registration import ImageRegistration

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


# Defines the decorator function for the log
def log(func):
    @wraps(func)
    def wrap(*args, **kwargs):
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f'function : {func.__name__} - execution time = {t1 - t0}s')
        return result
    return wrap


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


def get_h5_keys(folder, file_name, group=None):
    """ helper function used to analyze a h5 file and get a list of the keys.

    @param folder: (str) root folder where to look for the h5 file
    @param file_name: (str) name of the h5 file
    @param group: (str) indicate the name of the group to explore
    @return: (list) list of all the keys found in the file
    """
    if check_file_exist(folder, file_name):
        h5_data = h5py.File(os.path.join(folder, file_name), 'r')
        if group is not None:
            keys = list(h5_data[group].keys())
        else:
            keys = list(h5_data.keys())
    else:
        keys = []

    return keys


class DefineMask(ImageRegistration):

    def __init__(self, config_path):

        with open(config_path, "r") as config:
            param = yaml.load(config, Loader=yaml.FullLoader)

        # define the Ilastik folder as the main folder (required for Ilastik headless mode)
        self.Ilastik_folder: str = param['Ilastik_folder']
        os.chdir(self.Ilastik_folder)

        # load parameters regarding beta and alpha segmentation
        self.binning: bool = param['binning']
        self.rescaling_Z: str = param['rescaling_Z']

        # # load parameters for islet segmentation
        # self.islet_classification_model: str = param['islet_classification_model']

        # call the constructor for inherited classes
        self.registration: dict = param['registration_parameters']
        super().__init__(downsizing_power=self.registration['downsizing_power'],
                         gaussian_filter_size=self.registration['gaussian_filter_size'],
                         angles_range=self.registration['angles_range'],
                         zoom_range=self.registration['zoom_range'],
                         apply_zoom=self.registration['apply_zoom'],
                         verbose=False)

    def save_h5_image(self, im, name, dataset_path, update=False):
        """ Save image in hdf5 format for Ilastik

        @param update: (bool) indicate if the dataset is updated or newly created
        @param dataset_path: (str) indicate the dataset key associated to the saved image (key or group & key)
        @param im: (np array) stack of images to save
        @param name: (str) name of the h5 file in which the images will be saved
        """

        file_path = os.path.join(self.saving_path, name)

        if update:
            with h5py.File(file_path, "r+") as file:
                file[dataset_path][...] = im
        else:
            with h5py.File(file_path, "a") as file:
                file.create_dataset(dataset_path, data=im, dtype="uint16")

    @staticmethod
    def open_h5_image(path, name, dataset_path):
        """ Load an image or a stack of images from a h5 file, knowing group and key within the file.

        @param path: (str) path to the folder where the h5 file is saved
        @param name: (str) name of the h5 file
        @param dataset_path: (str) indicate the dataset key associated to the saved image
        @return: (nd array) loaded image
        """
        file_path = os.path.join(path, name)

        try:
            with h5py.File(file_path, 'r') as file:
                im = file[dataset_path]
                im = np.array(im)
        except OSError as err:
            print(err)
            raise

        return im

    def register_dapi_images(self, stack_reference, stack_to_align):
        """ Optimize registration based on the DAPI images and save the optimized parameters

        @param stack_to_align: (nd array) stack of images to register with respect to the reference stack
        @param stack_reference: (nd array) reference stack of images for registration
        """

        # Calculate the mip of the input images
        mip_reference = self.mip(stack_reference)
        mip_to_align = self.mip(stack_to_align)

        # Process the two images
        mip_reference_binned = self.bin_image(mip_reference)
        mip_reference_processed = self.process_image(mip_reference_binned, downsize=True)
        mip_to_align_binned = self.bin_image(mip_to_align)
        mip_to_align_processed = self.process_image(mip_to_align_binned, downsize=True)

        # Perform a first optimization, correcting rotation & shift
        self.optimize_rotation_translation(mip_reference_processed, mip_to_align_processed, downsize=True)

        # If the option was selected, perform a second optimization using a zoom morphological transformation. This
        # optimization is performed in 2 steps :
        # 1- the optimized shift and rotation are first applied
        # 2- the dilatation is then optimized by performing a grid search
        # 3- knowing the optimum dilatation, a new optimization of the rotation & translation is performed
        if self.apply_zoom:
            mip_registered = self.apply_2d_registration(mip_to_align_processed, downsize=True)
            self.optimize_dilatation(mip_reference_processed, mip_registered)
            mip_registered = self.im_dilate(mip_to_align_processed, self.registration_dic['zoom'])
            self.optimize_rotation_translation(mip_reference_processed, mip_registered, downsize=True)

        # Apply the transformation to the original image
        im_registered_final = self.apply_2d_registration(self.process_image(mip_to_align))
        self.save_aligned_overlay(mip_reference, im_registered_final, im_name="MIP")

        # Register the full stack
        binned_stack_to_align = self.bin_stack(stack_to_align)
        binned_stack_reference = self.bin_stack(stack_reference)
        stack_registered_to_align = self.apply_3d_registration(binned_stack_to_align, downsize=True)

        # Perform a final alignment along the z-axis
        self.optimize_3d_shift(binned_stack_reference, stack_registered_to_align, downsize=True)

        # Save the optimized parameters
        self.save_registration_dic()
        print(self.registration_dic)

    def load_manual_2d_mask(self, path, roi_folder, labels):
        """ This function is used to load a mask created manually using FIJI from a 2D image. The part of the image that
        was selected by the user has a pixel value equals to zero.

        @return:
        """

        # from the indicated path, look for the original image & the mask
        path_mip = os.path.join(path, roi_folder, "*ROI.tif")
        mip_file = glob(path_mip)
        path_mask = os.path.join(path, roi_folder, "*mask.tif")
        mask_file = glob(path_mask)

        # load the mip that was used to define the mask (it is only necessary to get the dimensions of the image in case
        # no mask was defined)
        if mip_file:
            im_mip = imread(mip_file)
            if self.binning:
                im_mip = self.bin_image(im_mip)
        else:
            print('error, the original image was not found.')
            return

        # load the mask or create it. If the mask exists, adapt the pixel value.
        if mask_file:
            im_mask = imread(mask_file)
            im_mask[im_mask == 0] = labels[0]
            im_mask[im_mask != 1] = labels[1]
            if self.binning:
                im_mask = self.bin_image(im_mask)
        else:
            im_mask = np.zeros(im_mip.shape) + labels[0]

        return im_mip, im_mask

    def save_data_for_ilastik(self, im, h5_filename, group_name, key):
        """ helper function to save the raw or formatted data that will be used later for ilastik segmentation.

        @param im: (np array) input image data (2d or 3d)
        @param h5_filename: (str) name of the h5 file where to save the data
        @param group_name: (str) indicate the name of the group under which the image will be saved
        @param key: (str) indicate the key used to save the data
        """
        keys = get_h5_keys(self.saving_path, h5_filename, group=group_name)
        if "raw_formatted" in keys:
            self.save_h5_image(im, h5_filename, group_name + "/" + key, update=True)
        else:
            self.save_h5_image(im, h5_filename, group_name + "/" + key)

    def format_stack_for_ilastik(self, stack, mip=False, bin_xy=False, rescale_z=None):
        """ Format the input stack before performing cell segmentation with Ilastik

        @param stack: (array) stack of images
        @param mip: (bool) indicate whether the mip should be computed
        @param bin_xy: (bool) indicate whether the image should be binned along xy
        @param rescale_z: (int) indicate by how much the stack should be rescale (removing frames)
        @return: (array) im, the formatted stack
        """
        if mip and bin_xy:
            im = self.mip(stack)
            im = self.bin_image(im)
        elif (not mip) and bin_xy:
            im = self.bin_stack(stack)
        else:
            im = stack

        if rescale_z is not None:
            im = self.reduce_frame_stack(im, rescale_z)

        return im

    # def load_beta_segmentation(self, roi_folder, mask_name):
    #
    #     path_mip = os.path.join(self.beta_classification_model, roi_folder, "*ROI.tif")
    #     mip = glob(path_mip)
    #     path_mask = os.path.join(self.beta_classification_model, roi_folder, "*mask.tif")
    #     mask = glob(path_mask)
    #
    #     # load the mip used to define the mask for the beta cells
    #     if mip:
    #        im_mip = imread(mip)
    #        if self.beta_model_apply_binning:
    #            im_mip = self.bin_image(im_mip)
    #     else:
    #        print('error')
    #        return
    #
    #     # load the mask or create it
    #     if mask:
    #         im_mask = imread(mask)
    #         im_mask[im_mask == 0] = 1
    #         im_mask[im_mask != 1] = 2
    #         if self.beta_model_apply_binning:
    #             im_mask = self.bin_image(im_mask)
    #     else:
    #         im_mask = np.ones(im_mip.shape)
    #
    #     # Save the formatted stack as h5
    #     keys = get_h5_keys(self.saving_path, "beta_stacks.h5", group="beta_rna")
    #     if "raw_formatted" in keys:
    #         self.save_h5_image(im_mip, "beta_stacks.h5", "beta_rna/raw_formatted", update=True)
    #     else:
    #         self.save_h5_image(im_mip, "beta_stacks.h5", "beta_rna/raw_formatted")
    #
    #     # Save the mask
    #     self.save_tiff_image(im_mask, os.path.join(self.saving_path, mask_name))

    def ilastik_segmentation(self, stack, mip, model, output_format, h5_filename, group_name, key):
        """ Launch segmentation with ilastik

        @param mip: (bool) indicate if a mip should be performed on the input image
        @param model: (str) full path pointing to the ilastik model file
        @param output_format: (str) indicate the format of the segmented image (e.g. probability map)
        @param h5_filename: (str) indicate the name of the h5 file where data for headless ilastik is saved
        @param group_name: (str) indicate the group name for the h5 file
        @param key:(str) key for h5 file
        @param stack: input stack
        """
        # Format the input stack according to the MIP and downsize options
        im = self.format_stack_for_ilastik(stack, mip=mip,
                                           bin_xy=self.binning,
                                           rescale_z=self.rescaling_Z)

        # Save the formatted stack as h5
        self.save_data_for_ilastik(im, h5_filename, group_name, key)

        # Launch the segmentation
        im_name = os.path.join(self.saving_path, h5_filename, group_name, key)
        ilastik_command = "./run_ilastik.sh --headless --project=" + model + \
                          " --export_source=" + output_format + " " + im_name
        os.system(ilastik_command)

        # return im

    # def ilastik_beta_segmentation(self, stack):
    #     """ Launch segmentation of the beta cells
    #
    #     @param stack: input stack
    #     """
    #     # Format the input stack according to the MIP and downsize options
    #     im = self.format_stack_for_ilastik(stack, mip=self.beta_model_apply_MIP,
    #                                        bin_xy=self.beta_model_apply_binning,
    #                                        rescale_z=self.beta_rescaling_Z)
    #
    #     # Save the formatted stack as h5
    #     keys = get_h5_keys(self.saving_path, "beta_stacks.h5", group="beta_rna")
    #     if "raw_formatted" in keys:
    #         self.save_h5_image(im, "beta_stacks.h5", "beta_rna/raw_formatted", update=True)
    #     else:
    #         self.save_h5_image(im, "beta_stacks.h5", "beta_rna/raw_formatted")
    #
    #     # Launch the segmentation
    #     im_name = os.path.join(self.saving_path, "beta_stacks.h5", "beta_rna", "raw_formatted")
    #     ilastik_command = "./run_ilastik.sh --headless --project=" + self.beta_classification_model + \
    #                       " --export_source='Simple Segmentation' " + im_name
    #     os.system(ilastik_command)

    # def ilastik_alpha_segmentation(self, stack):
    #     """ Launch segmentation of the alpha cells
    #
    #     @param stack: input stack
    #     """
    #     # Format the input stack according to the MIP and downsize options
    #     im = self.format_stack_for_ilastik(stack, mip=self.alpha_model_apply_MIP,
    #                                        bin_xy=self.alpha_model_apply_binning,
    #                                        rescale_z=self.alpha_rescaling_Z)
    #
    #     # Save the formatted stack as h5
    #     keys = get_h5_keys(self.saving_path, "alpha_stacks.h5", group="alpha_rna")
    #     if "raw_formatted" in keys:
    #         self.save_h5_image(im, "alpha_stacks.h5", "alpha_rna/raw_formatted", update=True)
    #     else:
    #         self.save_h5_image(im, "alpha_stacks.h5", "alpha_rna/raw_formatted")
    #
    #     # Launch the segmentation
    #     im_name = os.path.join(self.saving_path, "alpha_stacks.h5", "alpha_rna", "raw_formatted")
    #     ilastik_command = "./run_ilastik.sh --headless --project=" + self.alpha_classification_model + \
    #                       " --export_source='Probabilities' " + im_name
    #     os.system(ilastik_command)

    @staticmethod
    def reduce_frame_stack(stack, reducing_factor):
        """ Rescale the number of frames within a stack by a factor specified by the user

        @param stack: input stack of images
        @param reducing_factor: indicate by which factor the number of frames will be divided
        @return: reduced stack
        """
        lz, lx, ly = stack.shape
        reduced_frames = np.arange(start=0, stop=lz, step=reducing_factor)
        reduced_stack = np.zeros((len(reduced_frames), lx, ly))

        for n, frame in tqdm(enumerate(reduced_frames)):
            reduced_stack[n, :, :] = stack[frame, :, :]

        return reduced_stack

    def duplicate_frame_stack(self, stack, frames_number):
        """ Rescale the number of frames within a stack by a factor specified by the user

        @param frames_number: final number of frames composing the new stack
        @param stack: input stack of images
        @return: reduced stack
        """
        lz, lx, ly = stack.shape
        # if cell_type == "alpha":
        #     duplicating_factor = self.alpha_rescaling_Z
        # elif cell_type == "beta":
        #     duplicating_factor = self.beta_rescaling_Z
        # else:
        #     duplicating_factor = 1

        augmented_stack = np.zeros((lz * self.rescaling_Z, lx, ly))

        for frame in tqdm(range(lz)):
            for n in range(self.rescaling_Z):
                augmented_stack[self.rescaling_Z * frame + n, :, :] = stack[frame, :, :]

        if augmented_stack.shape[0] != frames_number:
            augmented_stack = augmented_stack[0:frames_number, :, :]

        return augmented_stack

    # def sort_traces_islet(self, trace_data, xy_shift, islet):
    #     """ Sort the traces based on the classes
    #
    #     @param xy_shift: (np array) values of the registration shift from pyHiM (DAPI vs. reference channel)
    #     @param beta: (np array) mask of the beta cells
    #     @param alpha: (np array) mask of the alpha cells
    #     @param trace_data: (pd dataframe) contains the trace data
    #     @return: (pd dataframe) return the modified dataframe after labelling
    #     """
    #     # create a new column called 'label'
    #     trace_data['label'] = ''
    #
    #     # defines the size of the alpha matrix
    #     #print(f'islet shape : {islet.shape}')
    #     islet = _sorting.bin_image(islet)
    #     lx, ly = islet.shape
    #
    #     scaling_factor = 2 ** self.downsizing_power  # * self.pixel_size[0]
    #     traces_x = (trace_data["x"] / self.pixel_size[0] - xy_shift[1]) / scaling_factor
    #     traces_y = (trace_data["y"] / self.pixel_size[0] - xy_shift[0]) / scaling_factor
    #     traces_z = trace_data["z"] / self.pixel_size[1]
    #
    #     # for each trace, calculate its average 3d position and associate a specific label based on the maks calculated
    #     # for the alpha and beta cells
    #     unique_trace_id = trace_data["Trace_ID"].drop_duplicates()
    #     for trace_id in tqdm(unique_trace_id):
    #
    #         index = trace_data.index[trace_data["Trace_ID"] == trace_id]
    #         x = np.mean(traces_x.loc[index])
    #         y = np.mean(traces_y.loc[index])
    #         z = np.mean(traces_z.loc[index])
    #
    #         # check if the trace is within the border of the images for the beta segmentation
    #         if (x > 0) and (x < lx) and (y > 0) and (y < ly):
    #             islet_label = islet[int(y), int(x)]
    #         else:
    #             islet_label = 0
    #
    #         # assign  a label to the selected trace
    #         if islet_label == 1:
    #             trace_label = 'islet'
    #         else:
    #             trace_label = 'exocrine'
    #
    #         for idx in index:
    #             trace_data.at[idx, 'label'] = trace_label
    #
    #     return trace_data
    #
    # def load_islet_segmentation(self, roi_folder, mask_name):
    #
    #     path_mask = os.path.join(self.islet_classification_model, roi_folder, "*mask.tif")
    #     mask = glob(path_mask)
    #
    #     # load the mask or create it
    #     if mask:
    #         im_mask = imread(mask)
    #         im_mask[im_mask == 0] = 1
    #         im_mask[im_mask != 1] = 0
    #
    #         # keys = get_h5_keys(self.saving_path, "islet_stacks.h5", group="islet_rna")
    #         # self.save_tiff_image(im_mask, os.path.join(self.saving_path, mask_name))
    #
    #         # im_mask[im_mask == 2] = 0
    #         # im_mask[im_mask > 0] = 1
    #         im_mask = ndimage.grey_opening(im_mask, size=(3, 3)).astype(int)
    #         im_mask = ndimage.binary_fill_holes(im_mask).astype(int)
    #
    #         # save the masks
    #         self.save_tiff_image(im_mask, "Islet_mask_manual_segmentation.tiff", rescale=False)
    #
    #     else:
    #         im_mask = np.zeros((2048, 2048))
    #         self.save_tiff_image(im_mask, "Islet_mask_manual_segmentation.tiff", rescale=False)
    #
    #     return im_mask