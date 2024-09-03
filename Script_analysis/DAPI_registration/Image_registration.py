# -*- coding: utf-8 -*-
"""
@authors: JB. Fiche
Created on Mon January 10, 2022
------------------------------------------------------------------------------------------------------------------------
This code is used to register two DAPI images, taken at two different time points of an experiment, after disassembly /
reassembly of the fluidics chamber. The mathematical process used below is image correlation.

From a terminal the script should be launched using the following command :
python Image_registration.py -F 'path_to_the_config_file'

For pycharm, the arguments are defined in Configurations/script parameters. If the configuration file is moved to
another location, the arguments must be updated.
------------------------------------------------------------------------------------------------------------------------
"""
import time

from tifffile import imread, TiffWriter
from scipy.signal import correlate
from scipy.ndimage import rotate, gaussian_filter, shift, zoom
from scipy.stats import pearsonr
from tqdm import tqdm
from functools import wraps
from time import time
from glob import glob
from argparse import ArgumentParser
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import json
import logging
import shutil
import yaml

matplotlib.use('TkAgg')
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


# def plot_images(im0, im):
#     plt.figure(figsize=(16, 10))
#     plt.imshow(im0, cmap='Blues')
#     plt.imshow(im, cmap='Reds', alpha=0.5)
#     plt.show()


class ImageRegistration:

    def __init__(self):

        arg = self.parse_argument()
        with open(arg.config_folder, "r") as configuration_file:
            parameters = yaml.load(configuration_file, Loader=yaml.FullLoader)

        registration_parameters = parameters['registration_parameters']
        self.downsizing_power: int = registration_parameters['downsizing_power']
        self.gaussian_filter_size: int = registration_parameters['gaussian_filter_size']
        self.angles_range: tuple = registration_parameters['angles_range']
        self.zoom_range: tuple = registration_parameters['zoom_range']
        self.apply_zoom: bool = registration_parameters['apply_zoom']
        self.crude_search_angle: int = np.ceil(np.sqrt((self.angles_range[1] - self.angles_range[0]) / 2))
        self.verbose: bool = registration_parameters['verbose']

        self.path_image_ref: str = parameters['reference_DAPI_folder']
        self.template_ref_file: str = parameters['template_reference_DAPI_file']
        self.ref_total_number_channels: int = parameters['number_channel_reference_stack']
        self.ref_channel: int = parameters['reference_channel']

        self.path_image_to_align: str = parameters['target_DAPI_folder']
        self.template_target_file: str = parameters['template_target_DAPI_file']
        self.align_total_number_channels: int = parameters['number_channel_target_stack']
        self.align_channel: int = parameters['target_channel']

        self.saving_path: str = os.path.join(parameters['dest_root_folder'],
                                             parameters['dest_analysis_folder'])
        self.overwrite_existing_file: bool = parameters['overwrite_existing_file']
        self.saving_folder_exist = self.create_saving_folder()

        self.registration_dic: dict = {"shift": [0, 0],
                                       "rotation": 0,
                                       "zoom": [1, 1],
                                       "z_shift": 0}

    @staticmethod
    def parse_argument():
        """ Parse the argument to retrieve the path to the configuration file.

        @return: parsed arguments.
        """
        parser = ArgumentParser()
        parser.add_argument("-F", "--config_folder",
                            type=str,
                            default=os.getcwd(),
                            help="Folder path for the configuration file")

        return parser.parse_args()

    def create_saving_folder(self):
        """ Check if the folder already exist. If it does, depending on the "overwrite_existing_file" option, the
        previous is removed or the analysis will be stopped.

        @return: (bool) True, the analysis can proceed. False, the analysis is aborted.
        """
        # Check if the folder already exist. and create the saving folder
        if os.path.isdir(self.saving_path) and self.overwrite_existing_file:
            shutil.rmtree(self.saving_path)
            os.mkdir(self.saving_path)
            return True
        elif os.path.isdir(self.saving_path) and not self.overwrite_existing_file:
            print('The saving path already exists. Analysis is aborted.')
            return False
        else:
            os.mkdir(self.saving_path)
            return True

    @staticmethod
    def load_stack(path, total_number_channels, selected_channel):
        """ Load images corresponding to a single channel from a multi channel stack.

        @param path: path to the selected stack
        @param total_number_channels: (int) total number of channels composing the stack
        @param selected_channel: (int) indicate the channel to extract
        @return: a stack of images corresponding to the selected channel
        """
        if len(path) == 1:
            im = imread(path[0])
            im = im[np.arange(start=selected_channel, stop=im.shape[0], step=total_number_channels, dtype=int), :, :]
            return im
        elif len(path) > 1:
            logger.warning('Load_stack was aborted since more than one path was detected.')
        else:
            logger.warning('Load_stack was aborted, no file was found.')

    @staticmethod
    def mip(stack):
        """ Calculate the maximum intensity projection of an input stack

        @param stack: input stack of images corresponding to a single acquisition channel
        @return: the mip as float
        """
        mip = np.max(stack, axis=0)
        mip = mip.astype(np.float_)
        return mip

    def bin_stack(self, stack):
        """ Apply binning to all images of a stack

        @param stack: input stack of images
        @return: binned stack
        """
        lz, lx, ly = stack.shape
        binning_factor = 2 ** self.downsizing_power
        binned_stack = np.zeros((lz, lx//binning_factor, ly//binning_factor))

        for frame in tqdm(range(lz)):
            binned_stack[frame, :, :] = self.bin_image(stack[frame, :, :])

        return binned_stack

    def bin_image(self, im):
        """ Bin (down-sizing) the input image in order to keep the essential structural information while saving time
        later during the alignment process. Indeed, the calculation for the correlation is much faster when working on
        small images.

        @param im: input 2d image
        @return: the binned image
        """
        # Bin the image according to the downsizing power indicated as parameter
        resize_shape = (int(im.shape[0] / 2 ** self.downsizing_power), int(im.shape[1] / 2 ** self.downsizing_power))

        shape = (resize_shape[0], im.shape[0] // resize_shape[0],
                 resize_shape[1], im.shape[1] // resize_shape[1])
        im_bin = im.reshape(shape).mean(-1).mean(1)

        return im_bin

    def process_image(self, im, downsize=False):
        """ Calculate the standardized image applying binning. The binning (down-sizing) is used to keep the essential
        structural information while saving time later during the alignment process. Indeed, the correlation is much
        faster when working on small images.

        @param downsize: indicate whether downsizing should be applied
        @param im: input 2d image
        @return: mip : maximum intensity projection of the original stack im
                 im_standardized : return the standardized mip after binning
        """
        # define the size of the gaussian kernel
        if downsize:
            filter_size = self.gaussian_filter_size / (2 * self.downsizing_power)
        else:
            filter_size = self.gaussian_filter_size

        # Correct for illumination inhomogeneity using a gaussian filter
        im_processed = np.divide(im, gaussian_filter(im, filter_size))

        # Standardized the image
        im_standardized = (im_processed - np.mean(im_processed)) / np.std(im_processed)

        return im_standardized

    @staticmethod
    def im_rotate(im, im_size, angle_value):
        """ Calculate the rotated image and crop the central part that is not affected by padding.

        @param im_size: size of the input image (assume to have the same x,y dimensions)
        @param im: input 2D image
        @param angle_value: value of the rotation
        @return: im_rotated, the cropped central part of the rotated image
        """

        # apply rotation by angle_value while keeping the same shape of the image
        im_rotated = rotate(im, angle_value, axes=(1, 0), reshape=False, order=1, mode='constant', prefilter=True)

        # select the central part of the rotated image (the one that will never be modified by padding whatever the
        # value of theta)
        d_shift = np.ceil((im_size - im_size / np.sqrt(2)) / 2)
        d_shift = d_shift.astype('int')
        im_rotated = im_rotated[d_shift:im_size - d_shift, d_shift:im_size - d_shift]

        return im_rotated

    def angle_search(self, angles, n_angles, roi, roi_shape, im_ref):
        """ For each angle, perform a rotation of an ROI selected on the image we want to register. Correlate it to the
        reference image.

        @param angles: (ndarray) containing all the values to test
        @param n_angles: (int) number of angles to test
        @param roi: (2d ndarray) central roi selected on the 2d image we need to register
        @param roi_shape: (2d tuple) size of roi
        @param im_ref: (2d ndarray) reference image for the registration
        @return: correlation_max : (1d ndarray) maximum value for the correlation for each value of angles
                 max_correlation_2d_pos : (2d ndarray) x,y position of the pixel with the highest correlation value. It
                 will be used later to indicated by how much the image to register should be shifted with respect to the
                 reference.
        """
        correlation_max = np.zeros((n_angles,))
        correlation_im = np.zeros((n_angles, im_ref.shape[0], im_ref.shape[1]))
        max_correlation_2d_pos = np.zeros((n_angles, 2))

        # For each angle, rotate the roi and calculate the correlated image with the reference
        for n, angle in enumerate(angles):
            roi_rotated = self.im_rotate(roi, roi_shape, angle)
            corr = correlate(im_ref, roi_rotated, mode='same')
            correlation_max[n] = np.max(corr)
            correlation_im[n] = corr
            max_correlation_2d_pos[n, :] = np.unravel_index(np.argmax(corr, axis=None), corr.shape)

        # Plot the correlated images if the verbose option was selected
        if self.verbose:
            self.plot_correlation(correlation_im, angles)

        return correlation_max, max_correlation_2d_pos

    @log
    def optimize_rotation_translation(self, im_ref, im, downsize=False):
        """ Perform a search procedure to align an input image im with respect to im_ref. The aim is to find the
        rotation angle leading to the highest correlation score for the two images. The procedure works in two steps :
        first a crude search is performed according to the specified parameters. Then, a finer search is run, with an
        angle step of 1°.

        @param downsize: indicate whether the image was binned before being processed
        @param im_ref: reference 2D input image
        @param im: image to be aligned
        @return: the optimum angle value (returning the highest correlation value) as well as the optimum shift values
        """
        # Select the central roi of the image that needs to be realigned
        roi, x0, y0 = self.select_roi(im)
        roi_shape = roi.shape[0]

        # Calculate the multiplicative factor for the translation
        if downsize:
            downsizing_factor = 2 ** self.downsizing_power
        else:
            downsizing_factor = 1

        # Optimization is performed in two steps - the first step consists in a crude search to narrow down the optimum
        # angle for the rotation
        n_angles = (self.angles_range[1] - self.angles_range[0]) / self.crude_search_angle
        n_angles = np.ceil(n_angles).astype('uint8')
        angles_crude_search = np.linspace(self.angles_range[0], self.angles_range[1], num=int(n_angles),
                                          endpoint=False,
                                          retstep=False,
                                          dtype=None, axis=0)
        correlation_max, max_correlation_2d_pos = self.angle_search(angles_crude_search, n_angles, roi, roi_shape,
                                                                    im_ref)

        # A second search is performed between the bounds defined earlier with a step of 1°
        starting_angle = np.round(angles_crude_search[np.argmax(correlation_max)])
        angles_fine_search = np.arange(starting_angle - (self.crude_search_angle - 1),
                                       starting_angle + self.crude_search_angle, 1)
        n_angles = len(angles_fine_search)
        correlation_max, max_correlation_2d_pos = self.angle_search(angles_fine_search, n_angles, roi, roi_shape,
                                                                    im_ref)

        # Return the angle with the best correlation score as well as the shift dx, dy values - update the registration
        # dictionary
        max_correlation_idx = np.argmax(correlation_max)
        self.registration_dic["rotation"] = angles_fine_search[max_correlation_idx]
        dx = (x0 - max_correlation_2d_pos[max_correlation_idx, 0]) * downsizing_factor
        dy = (y0 - max_correlation_2d_pos[max_correlation_idx, 1]) * downsizing_factor
        self.registration_dic["shift"] = [dx, dy]

        return np.max(correlation_max)

    @staticmethod
    def im_dilate(im, zoom_factor):
        """ Perform dilation on input image according to the specified zoom factors. The output image must have
        the same size that the input image. Therefore, if zoom-factor > 1, the output image is cropped accordingly. If
        zoom-factor < 1, zero padding is applied to the output image.

        @param im: input image on which dilatation is applied
        @param zoom_factor: tuple containing the dilation factors along X and Y
        @return: output image after applying dilatation. It has the same size that the input image im.
        """
        dilated_im = zoom(im, zoom=zoom_factor)
        lx, ly = im.shape
        lx_zoom, ly_zoom = dilated_im.shape

        if (lx_zoom >= lx) and (ly_zoom >= ly):
            dilated_im = dilated_im[int((lx_zoom - lx) / 2):int((lx_zoom - lx) / 2) + lx,
                                    int((ly_zoom - ly) / 2):int((ly_zoom - ly) / 2) + ly]
        else:
            template = np.zeros((lx, ly))
            template[int((lx - lx_zoom) / 2):int((lx - lx_zoom) / 2) + lx_zoom,
                     int((ly - ly_zoom) / 2):int((ly - ly_zoom) / 2) + ly_zoom] = dilated_im
            dilated_im = template

        if dilated_im.shape != im.shape:
            print(f'lx={lx}, ly={ly}')
            print(f'lx_zoom={lx_zoom}, ly_zoom={ly_zoom}')

        return dilated_im

    def optimize_dilatation(self, im_ref, im):
        """ Perform zoom transformation to optimize image registration. Since the optimum transformation is not always
        symmetric, the zoom factor is optimize in 2D using a grid search algorithm (step of 2%). The best parameters are
        saved in the registration dictionary.

        @param im_ref: reference image to use for registration
        @param im: im to modify to optimize the registration
        @return: the best combination of zoom parameters
        """
        dilatation_range = np.arange(self.zoom_range[0], self.zoom_range[1], 0.005)
        correlation_max = np.zeros((dilatation_range.shape[0], dilatation_range.shape[0]))

        # For each dilatation factor, perform a dilatation of the image and calculate the correlation
        for nx, zoom_factor_x in tqdm(enumerate(dilatation_range)):
            for ny, zoom_factor_y in enumerate(dilatation_range):
                zoom_factor = (zoom_factor_x, zoom_factor_y)
                dilated_im = self.im_dilate(im, zoom_factor)

                roi, x0, y0 = self.select_roi(dilated_im)
                roi_shape = roi.shape[0]
                corr, _ = self.angle_search(np.arange(-1, 2, 1), 3, roi, roi_shape, im_ref)
                correlation_max[nx, ny] = np.max(corr)

        # Return the dilatation factor with the best correlation score as well as the shift dx, dy values. Update the
        # registration dictionary
        max_correlation_idx = np.unravel_index(np.argmax(correlation_max),
                                               (len(dilatation_range), len(dilatation_range)))
        self.registration_dic["zoom"] = [dilatation_range[max_correlation_idx[0]],
                                         dilatation_range[max_correlation_idx[1]]]

        return np.max(correlation_max)

    @staticmethod
    def select_roi(im):
        """ Select the central roi of the input image.

        @param im: input 2D image
        @return: roi : the central part of the input image
        """
        lx, ly = im.shape
        x0 = int(lx / 2) - 1
        y0 = int(ly / 2) - 1
        d = int(lx / 2)
        roi = im[x0 - d // 2:x0 + d // 2, y0 - d // 2:y0 + d // 2]
        # roi = (roi - np.mean(roi)) / np.std(roi)

        return roi, x0, y0

    @log
    def optimize_3d_shift(self, stack_ref, stack, downsize=False):
        """ Perform 3d registration based on correlation. Similar to 2d registration, the central 3d roi of the stack we
        need to register is selected and correlated to the reference stack. For this method to work, it is important to
        work with normalized images.


        @param stack_ref: reference 3d stack of images
        @param stack: 3d stack of images to register - a first step registration should have been applied already for
        correcting rotation, shift and zoom.
        @param downsize: (bool) indicate whether downsizing was applied
        """
        # normalize stacks - each image of the stack is normalized separately to make sure the background is properly
        # removed. This is an important step, in order to avoid any bias induced by the variation of fluorescence
        # intensity within the stack. Note that the two stacks are sometimes of different shape.
        lz, lx, ly = stack.shape
        for frame in range(lz):
            stack[frame, :, :] = self.process_image(stack[frame, :, :], downsize=downsize)

        lz, lx, ly = stack_ref.shape
        for frame in range(lz):
            stack_ref[frame, :, :] = self.process_image(stack_ref[frame, :, :], downsize=downsize)

        # Select the central roi of the image that needs to be realigned
        roi_stack, x0, y0, z0 = self.select_roi_stack(stack)

        # Calculate 3d correlation of the two stacks, keeping the reference stack with its original size
        corr = correlate(stack_ref, roi_stack, mode='same')
        max_pos = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
        dz = z0 - max_pos[0]

        # save dz
        self.registration_dic["z_shift"] = float(dz)

    @staticmethod
    def select_roi_stack(stack):
        """ Select the central roi of the input image.

        @param stack: input 3d stack of images
        @return: roi : the central part of the input stack
        """
        lz, lx, ly = stack.shape
        z0 = int(lz / 2) - 1
        x0 = int(lx / 2) - 1
        y0 = int(ly / 2) - 1
        dz = int(lz / 2)
        dxy = int(lx / 2)

        roi = stack[z0 - dz // 2:z0 + dz // 2,
                    x0 - dxy // 2:x0 + dxy // 2,
                    y0 - dxy // 2:y0 + dxy // 2]
        #roi = (roi - np.mean(roi)) / np.std(roi)

        return roi, x0, y0, z0

    def apply_2d_registration(self, im, downsize=False):
        """ Apply the alignment transformation to the input 2D image (not binned)

        @param downsize: indicate whether the input image was binned
        @param im: 2D image (np array)
        @return: im_registered, the modified image.
        """

        # Registration dictionary
        optimum_angle = self.registration_dic["rotation"]
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # If the selected image was binned, adapt the shift corrections
        if downsize:
            downsizing_factor = 2 ** self.downsizing_power
            dx = dx / downsizing_factor
            dy = dy / downsizing_factor

        # Perform registration according to the parameters
        im_rotated = rotate(im, optimum_angle, axes=(1, 0), reshape=False, order=1, mode='constant', cval=0.0)
        im_rotated_shifted = shift(im_rotated, [-dx, -dy], order=0, mode='constant')
        im_rotated_shifted_zoom = self.im_dilate(im_rotated_shifted, optimum_zoom)

        return im_rotated_shifted_zoom

    def apply_3d_registration(self, stack, downsize=False):
        """ Recalculate the input stack after applying registration

        @param downsize: indicate whether binning was applied to the input stack
        @param stack: stack to which registration will be applied
        @return: return the registered stack
        """
        lz, lx, ly = stack.shape
        aligned_stack_2d = np.zeros((lz, lx, ly))

        # for each image, apply the 2d registration optimized for the dataset
        for frame in tqdm(range(lz)):
            aligned_stack_2d[frame, :, :] = self.apply_2d_registration(stack[frame, :, :], downsize=downsize)

        # apply the z shift
        dz = self.registration_dic["z_shift"]
        aligned_stack_3d = shift(aligned_stack_2d, [-dz, 0, 0], order=0, mode='constant')

        return aligned_stack_3d

    def save_tiff_image(self, im, name, rescale=False):
        """ Save input image as a tiff file.

        @param rescale: boolean - indicate whether the intensity should be rescaled before saving the tif
        @param im: input 2D images
        @param name: saving name
        """
        im_path = os.path.join(self.saving_path, name)
        if rescale:
            im_min = np.min(im)
            im_max = np.max(im)
            im = (2 ** 16 - 1) * (im - im_min) / (im_max - im_min)

        with TiffWriter(im_path) as tif:
            tif.write(im.astype(np.uint16))

    def save_registration_dic(self, prefix=''):
        """ Save in a json file the dictionary containing the optimized parameters for registration
        """
        registration_file = os.path.join(self.saving_path, f'{prefix}registration_parameters.json')
        with open(registration_file, 'w') as file:
            json.dump(self.registration_dic, file)

    def load_registration_dic(self, folder):
        """ Load the json file containing the dictionary with the optimized parameters for registration

        @param folder: folder where to look for file where the registration parameters were saved
        """
        file_path = os.path.join(folder, 'registration_parameters.json')
        f = open(file_path)
        self.registration_dic = json.load(f)

    @staticmethod
    def plot_correlation(im, angles):
        """ Plot all the correlated images if the verbose option is selected.

        @param im: array of images
        @param angles: array of angles tested for the correlation
        """
        plt.figure(figsize=(16, 10))
        num_im = im.shape[0]

        subplot_x = np.round(np.sqrt(num_im))
        subplot_y = np.ceil(num_im / subplot_x)

        for n in range(num_im):
            plt.subplot(subplot_x, subplot_y, n + 1)
            plt.imshow(im[n], cmap='gray')
            plt.title(f'{angles[n]}°')
            plt.axis('off')

        plt.show()

    def rescale_contrast(self, im):
        """ This function is used to improve the contrast of an input image for the png images. The input image is first
        processed and the intensity is rescaled in a second step.

        @param im: input 2D image (np array)
        @return: 2D image with rescaled intensity
        """
        im = np.divide(im, gaussian_filter(im, self.gaussian_filter_size))
        intensity = np.copy(im)
        intensity = np.reshape(intensity, (intensity.shape[0] * intensity.shape[1], 1))
        intensity = np.sort(intensity, axis=0)
        int_min = np.percentile(intensity, 0.5)
        int_max = np.percentile(intensity, 99.5)
        im = (2 ** 16 - 1) * (im - int_min) / (int_max - int_min)
        im[im < 0] = 0
        im[im > 2 ** 16 - 1] = 2 ** 16
        return im

    def save_aligned_overlay(self, im_ref, im_aligned, im_name="MIP"):
        """ Overlay the registered image with the reference and save the result

        @param im_ref: reference 2D image
        @param im_aligned: aligned image
        @param im_name: name of the image to save
        """

        # Registration dictionary
        optimum_angle = np.around(self.registration_dic["rotation"], decimals=1)
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # Rescale contrast for the original images
        im_ref_rescaled = self.rescale_contrast(im_ref)

        # Save the aligned montage :
        im_title = im_name + '_registered_montage.png'
        im_path = os.path.join(self.saving_path, im_title)
        plt.figure(figsize=(16, 10))
        plt.imshow(im_ref_rescaled, cmap='Blues')
        plt.imshow(im_aligned, cmap='Reds', alpha=0.5)
        plt.title(f'Aligned images - angle {optimum_angle}° - shift {dx, dy}px - zoom {optimum_zoom}')
        plt.axis('off')
        plt.savefig(im_path)

    def save_aligned_montage(self, im_ref, im_2_align, im_aligned, im_name="MIP"):
        """ Save the results

        @param im_ref: reference 2D image
        @param im_2_align: image to align
        @param im_aligned: aligned image
        @param im_name: name of the image to save
        """
        plt.figure(figsize=(16, 10))

        # Registration dictionary
        optimum_angle = np.around(self.registration_dic["rotation"], decimals=1)
        dx, dy = self.registration_dic["shift"]
        optimum_zoom = self.registration_dic["zoom"]

        # Rescale contrast for the original images
        im_ref_rescaled = self.rescale_contrast(im_ref)
        im_2_align_rescaled = self.rescale_contrast(im_2_align)

        # Save the images as a subplot
        plt.subplot(131)
        plt.imshow(im_ref_rescaled, cmap='gray')
        plt.title('Reference')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(im_2_align_rescaled, cmap='gray')
        plt.title('Original image to align')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(im_ref_rescaled, cmap='Blues')
        plt.imshow(im_aligned, cmap='Reds', alpha=0.5)
        plt.title(f'Aligned images - angle {optimum_angle}° - shift {dx, dy}px - zoom {optimum_zoom}')
        plt.axis('off')

        im_title = im_name + '_registered.png'
        im_path = os.path.join(self.saving_path, im_title)
        plt.savefig(im_path)

    def simulate_data(self, stack, shift_range, angle_range, zoom_range, shift_3d_range, output_name):

        # from the original stack, remove the 5 first and last frames (for the simulations of the z-shift)
        lz = stack.shape[0]
        reduced_stack = stack[shift_3d_range: lz - shift_3d_range, :, :]

        # from the parameters, randomly create a set of transformations
        im_zoom = np.random.uniform(low=zoom_range[0], high=zoom_range[1], size=(2,))
        # im_zoom = np.random.uniform(low=zoom_range[0], high=zoom_range[1])
        im_shift = np.random.uniform(low=-shift_range, high=shift_range, size=(2,))
        im_rotation = np.random.uniform(low=-angle_range, high=angle_range)
        im_shift_3d = np.random.uniform(low=-shift_3d_range, high=shift_3d_range)

        # save the parameters in the registration dictionary.
        self.registration_dic: dict = {"shift": list(np.round(im_shift)),
                                       "rotation": np.round(im_rotation),
                                       # "zoom": [np.around(im_zoom, decimals=2), np.around(im_zoom, decimals=2)],
                                       "zoom": list(np.around(im_zoom, decimals=2)),
                                       "z_shift": np.round(im_shift_3d)}
        print(self.registration_dic)

        # Save the registration dictionary as a json file
        self.save_registration_dic(prefix='simulated_')

        # apply the simulated parameters to the input stack and save the result. The z-shift is set to zero. It will be
        # applied manually later in order to avoid the creation of empty frames
        self.registration_dic["z_shift"] = 0
        simulated_stack = self.apply_3d_registration(stack, downsize=False)

        # apply the z-shift to the simulated stack
        simulated_stack = simulated_stack[int(shift_3d_range + np.round(im_shift_3d)):
                                          int(lz - shift_3d_range + np.round(im_shift_3d)), :, :]

        # # replace all pixel equal to zero by a random noise typical for the sCMOS
        # lz, lx, ly = simulated_stack.shape
        # simulated_stack = np.reshape(simulated_stack, (lz * lx * ly, ))
        # idx = np.argwhere(simulated_stack == 0)
        # noise = np.random.normal(loc=10, scale=1, size=(idx.shape[0],)) + \
        #         np.random.normal(loc=100, scale=1.2, size=(idx.shape[0],))
        # simulated_stack[idx[:, 0]] = noise
        # simulated_stack = np.reshape(simulated_stack, (lz, lx, ly))

        # replace all pixels equal to zero by 1
        simulated_stack[simulated_stack == 0] = 1

        # reset the registration dictionary
        self.registration_dic: dict = {"shift": [0, 0],
                                       "rotation": 0,
                                       "zoom": [1, 1],
                                       "z_shift": 0}

        # save the simulated stack
        # self.save_tiff_image(simulated_stack, output_name, rescale=False)
        return reduced_stack, simulated_stack

    def run_registration(self, downsize=False):
        """ Run the full registration pipeline

        @param downsize: (bool) indicate whether the analysis will be performed on the raw image or a binned image
        """
        if not self.saving_folder_exist:
            return

        # define the path to the images
        path_ref = glob(os.path.join(self.path_image_ref, self.template_ref_file))
        path_to_align = glob(os.path.join(self.path_image_to_align, self.template_target_file))

        # Load the reference image
        stack_reference = self.load_stack(path_ref, self.ref_total_number_channels, self.ref_channel)
        stack_to_align = self.load_stack(path_to_align, self.align_total_number_channels, self.align_channel)

        # Calculate the MIP for the two images
        mip_reference = self.mip(stack_reference)
        mip_to_align = self.mip(stack_to_align)

        Lx, Ly = mip_reference.shape
        # _align.save_tiff_image(mip_reference, "MIP_ref")
        # _align.save_tiff_image(mip_to_align, "MIP_to_align")

        # Process the two images
        mip_to_align_binned = self.bin_image(mip_to_align)
        mip_to_align_processed = self.process_image(mip_to_align_binned, downsize=downsize)
        mip_reference_binned = self.bin_image(mip_reference)
        mip_reference_processed = self.process_image(mip_reference_binned, downsize=downsize)

        # Perform a first optimization, correcting rotation & shift
        _align.optimize_rotation_translation(mip_reference_processed, mip_to_align_processed, downsize=downsize)

        # If the option was selected, perform a second optimization using a zoom morphological transformation. This
        # optimization is performed in 2 steps :
        # 1- the optimized shift and rotation are first applied
        # 2- the dilatation is then optimized by performing a grid search
        # 3- knowing the optimum dilatation, a new optimization of the rotation & translation is performed
        if _align.apply_zoom:
            mip_registered = self.apply_2d_registration(mip_to_align_processed, downsize=downsize)
            self.optimize_dilatation(mip_reference_processed, mip_registered)
            mip_registered = self.im_dilate(mip_to_align_processed, _align.registration_dic['zoom'])
            self.optimize_rotation_translation(mip_reference_processed, mip_registered, downsize=downsize)

        # Apply the transformation to the original image
        im_registered_final = self.apply_2d_registration(_align.process_image(mip_to_align))

        # Plot and save the results for registration based only on rotation and shift morphological transformations
        self.save_aligned_montage(mip_reference, mip_to_align, im_registered_final, im_name="MIP_registered")
        self.save_aligned_overlay(mip_reference, im_registered_final, im_name="MIP_registered")

        # Register the full stack
        binned_stack_to_align = self.bin_stack(stack_to_align)
        binned_stack_reference = self.bin_stack(stack_reference)
        stack_registered = self.apply_3d_registration(binned_stack_to_align, downsize=downsize)

        # Perform a final alignment along the z-axis
        self.optimize_3d_shift(binned_stack_reference, stack_registered, downsize=downsize)

        # Save the registration dic
        self.save_registration_dic()
        print(self.registration_dic)

        # Final registration of the stack and saving
        stack_registered_final = self.apply_3d_registration(stack_to_align, downsize=False)
        self.save_tiff_image(stack_registered_final, "final_registered_stack", rescale=False)


if __name__ == "__main__":

    # Instantiate the alignment class
    _align = ImageRegistration()

    # Perform the registration
    _align.run_registration(downsize=True)


    # for n_simu in range(50):
    #
    #     # Instantiate the alignment class
    #     _align = ImageRegistration(downsizing_power=1, gaussian_filter_size=10, angles_range=(-21, 21),
    #                                zoom_range=(1, 1.15), apply_zoom=True, verbose=False)
    #
    #     # Define and create the saving folder
    #     simulation_folder = f'{str(n_simu).zfill(3)}_simulation'
    #     _align.saving_path = os.path.join(dest_folder, simulation_folder)
    #     if os.path.isdir(_align.saving_path):
    #         shutil.rmtree(_align.saving_path)
    #     os.mkdir(_align.saving_path)
    #
    #     # Load the reference image
    #     stack_reference = _align.load_stack(path_image_ref, ref_total_number_channels, ref_channel)
    #     # stack_to_align = _align.load_stack(path_image_to_align, align_total_number_channels, align_channel)
    #
    #     # Create the simulated image by randomly defining the transformation parameters
    #     stack_reference, stack_to_align = _align.simulate_data(stack_reference, 50, 20, [0.9, 1], 5, 'simulation.tif')
    #
    #     # Calculate the MIP for the two images
    #     mip_reference = _align.mip(stack_reference)
    #     mip_to_align = _align.mip(stack_to_align)
    #
    #     Lx, Ly = mip_reference.shape
    #     # _align.save_tiff_image(mip_reference, "MIP_ref")
    #     # _align.save_tiff_image(mip_to_align, "MIP_to_align")
    #
    #     # Process the two images
    #     mip_to_align_binned = _align.bin_image(mip_to_align)
    #     mip_to_align_processed = _align.process_image(mip_to_align_binned, downsize=True)
    #     mip_reference_binned = _align.bin_image(mip_reference)
    #     mip_reference_processed = _align.process_image(mip_reference_binned, downsize=True)
    #
    #     # # mip_to_align_binned = _align.bin_image(mip_to_align)
    #     # mip_to_align_processed = _align.process_image(mip_to_align, downsize=False)
    #     # # mip_reference_binned = _align.bin_image(mip_reference)
    #     # mip_reference_processed = _align.process_image(mip_reference, downsize=False)
    #
    #     # Perform a first optimization, correcting rotation & shift
    #     _align.optimize_rotation_translation(mip_reference_processed, mip_to_align_processed, downsize=True)
    #
    #     # If the option was selected, perform a second optimization using a zoom morphological transformation. This
    #     # optimization is performed in 2 steps :
    #     # 1- the optimized shift and rotation are first applied
    #     # 2- the dilatation is then optimized by performing a grid search
    #     # 3- knowing the optimum dilatation, a new optimization of the rotation & translation is performed
    #     if _align.apply_zoom:
    #         mip_registered = _align.apply_2d_registration(mip_to_align_processed, downsize=True)
    #         _align.optimize_dilatation(mip_reference_processed, mip_registered)
    #         mip_registered = _align.im_dilate(mip_to_align_processed, _align.registration_dic['zoom'])
    #         _align.optimize_rotation_translation(mip_reference_processed, mip_registered, downsize=True)
    #
    #     # Apply the transformation to the original image
    #     im_registered_final = _align.apply_2d_registration(_align.process_image(mip_to_align))
    #
    #     # Plot and save the results for registration based only on rotation and shift morphological transformations
    #     _align.save_aligned_montage(mip_reference, mip_to_align, im_registered_final, im_name="MIP_registered")
    #     _align.save_aligned_overlay(mip_reference, im_registered_final, im_name="MIP_registered")
    #
    #     # Register the full stack
    #     binned_stack_to_align = _align.bin_stack(stack_to_align)
    #     binned_stack_reference = _align.bin_stack(stack_reference)
    #     stack_registered = _align.apply_3d_registration(binned_stack_to_align, downsize=True)
    #
    #     # Perform a final alignment along the z-axis
    #     _align.optimize_3d_shift(binned_stack_reference, stack_registered, downsize=True)
    #
    #     # Final registration of the stack and saving
    #     stack_registered_final = _align.apply_3d_registration(stack_to_align, downsize=False)
    #     _align.save_tiff_image(stack_registered_final, "final_registered_stack", rescale=False)
    #
    #     # Calculate the correlation, taking into account only the central part of the image
    #     print('Calculate final correlations ...')
    #
    #     nframes = stack_reference.shape[0]
    #     correlation = np.zeros((nframes, 2))
    #     for nframe in tqdm(range(nframes)):
    #         stack_reference_roi = stack_reference[nframe, 512: 1536, 512: 1536]
    #         stack_reference_roi = stack_reference_roi.reshape((1024 * 1024,))
    #         stack_registered_roi = stack_registered_final[nframe, 512: 1536, 512: 1536]
    #         stack_registered_roi = stack_registered_roi.reshape((1024 * 1024,))
    #         correlation[nframe, :] = pearsonr(stack_reference_roi, stack_registered_roi)
    #
    #     np.save(os.path.join(_align.saving_path, 'correlation.npy'), correlation)
    #
    #     # Save the final parameters :
    #     print(_align.registration_dic)
    #     _align.save_registration_dic()
