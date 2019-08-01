import SimpleITK as sitk
import os
import numpy as np
from scipy import ndimage
import scipy
from skimage.exposure import equalize_adapthist
import configuration
import os
from os.path import isfile, join
from os import listdir
import matplotlib.pyplot as plt
import cv2
import random
import matplotlib.gridspec as gridspec
import scipy.stats
import pandas as pd
import pickle
from skimage import measure
from skimage import exposure
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
#%%
#########
#
########
def SaveTheResult(data, filename):
    print('Proc: Saving the result into %s....' %(filename))
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print('Done: Saved the result....')

def LoadtheResult(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def GetLastWeight(directory, prefix="weights", ends=".h5"):
        files = [os.path.splitext(os.path.basename(i))[0] for i in os.listdir(directory) if os.path.isfile(os.path.join(directory,i)) and prefix in i and i.endswith(ends)]
        if (len(files)==0):
            return None
        best_epoch = 1
        weight_epochs = []
        for fil in files:
            weight_epochs.append(int(fil.split("-")[1]))
        best_epoch = max(weight_epochs)
        print(best_epoch)
        #Construct the file name
        file_name_weight = directory + '/weights-%02d.h5' % (best_epoch)
        return file_name_weight

#########
#
# 3D Viewer
#
###########
def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

def CorrectN4Bias(img, verbose=False):
    # input img must be sitkFloat32 or sitkFloat64 data type
    maskImage = sitk.OtsuThreshold(img, 0, 1, 200)
    inputImage = sitk.Cast(img,sitk.sitkFloat32)
    # inputImage = sitk.Shrink( inputImage, 0.5 * inputImage.GetDimension() )
    # maskImage = sitk.Shrink( maskImage, 0.5 * inputImage.GetDimension() )
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # corrector.SetMaximumNumberOfIterations(np.uint32(5))
    output = corrector.Execute(inputImage, maskImage)
    # output  = sitk.Expand(output, 2 * output.GetDimension())
    if verbose:
        o =sitk.GetArrayFromImage(output)
        multi_slice_viewer(o,o)
        plt.show()
    return output

def smooth_contours(img, verbose=False, sigma=5.0, threshold=0.35, type='Classic'):
    if (type == 'Classic'):
        img_reshape = np.float32(img.reshape(configuration.standard_volume))
        img_gaussian = np.zeros(configuration.standard_volume)
        output = np.zeros(configuration.standard_volume)
        for i in range(img_reshape.shape[0]):
            img_gaussian[i] = gaussian_filter(img_reshape[i],sigma=sigma)
            for j in range(img_reshape.shape[1]):
                for k in range(img_reshape.shape[2]):
                    output[i][j][k] = (img_gaussian[i][j][k] > threshold)
        return output
    elif (type == 'CV2'):
        img_reshape = img.reshape(configuration.standard_volume)
        img_erosion = np.zeros(configuration.standard_volume)
        output = np.zeros(configuration.standard_volume)
        kernel = np.ones((3,3),np.uint8)
        for i in range(img_reshape.shape[0]):
            img_erosion[i] = cv2.erode(img_reshape[i],kernel,iterations=1)
            output[i] = cv2.dilate(img_erosion[i],kernel,iterations=1)
        return output

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """
    imgs_x = sitk.GetImageFromArray(imgs)
    blurFilter = sitk.CurvatureFlowImageFilter()
    blurFilter.SetNumberOfIterations(n_iter)
    blurFilter.SetTimeStep(t_step)
    imgs_x = blurFilter.Execute(imgs_x)
    imgs_x = sitk.GetArrayFromImage(imgs_x)
    return imgs_x


def multi_slice_viewer_legacy(volume, img, depth=1):
    # volume is the prediction, img is the MRI input
    remove_keymap_conflicts({'j', 'k', 'left', 'right'})
    img_mask = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], 3))

    for i, x in enumerate(img):
        if depth == 1:
            for j in range(3):
                img_mask[i, :, :, j] = x
            mask = volume[i] == 1
        else:
            img_mask[i, :, :, j] = x
            if volume.shape[3] == 3:
                mask = np.logical_or(volume[i, :, :, 2] >= 0.43, volume[i, :, :, 1] > 0.43)

        mask_emp = np.zeros((x.shape[0], x.shape[0]), dtype=np.uint8)
        mask_emp[mask] = 255
        contours, hierarchy = cv2.findContours(mask_emp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours):
            cv2.drawContours(img_mask[i], contours, -1, (1, 0, 0), 1)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].volume = volume
    ax[0].index = volume.shape[0] - 1  # // 2
    ax[0].imshow(volume[ax[0].index], vmin=0.0, vmax=1.0)
    ax[1].volume = img_mask
    ax[1].index = img_mask.shape[0] - 1  # // 2
    ax[1].imshow(img_mask[ax[1].index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def multi_slice_viewer(volume, reference, img, depth=1):
    #volume is the prediction, reference is the expert segmentation, img is the MRI input
    remove_keymap_conflicts({'j', 'k','left','right'})
    img_mask = np.zeros((volume.shape[0],volume.shape[1],volume.shape[2],3))

    for i, x in enumerate(img):
        if depth==1:
            for j in range(3):
                img_mask[i,:,:,j] = x
            mask = volume[i] == 1
            mask2 = reference[i] == 1
        else:
            img_mask[i,:,:,j] = x
            if volume.shape[3]==3:
                mask = np.logical_or(volume[i,:,:,2]>=0.43,volume[i,:,:,1]>0.43)
   
        mask_emp = np.zeros((x.shape[0],x.shape[0]), dtype=np.uint8)
        mask_emp[mask] = 255
        ref_emp = np.zeros((x.shape[0],x.shape[0]), dtype=np.uint8)
        ref_emp[mask2] = 255
        contours, hierarchy = cv2.findContours(mask_emp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours_ref, hierarchy = cv2.findContours(ref_emp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours_ref):
            cv2.drawContours(img_mask[i], contours_ref, -1, (0,1,0), 1)

        if len(contours):
            cv2.drawContours(img_mask[i], contours, -1, (1,0,0), 1)
        
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].volume = volume
    ax[0].index = volume.shape[0] -1 #// 2
    ax[0].imshow(volume[ax[0].index],vmin=0.0,vmax=1.0)
    ax[1].volume = img_mask
    ax[1].index = img_mask.shape[0] -1 #// 2
    ax[1].imshow(img_mask[ax[1].index])
    fig.canvas.mpl_connect('key_press_event', process_key)

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    ab = fig.axes[1]
    if event.key == 'j' or event.key == 'left':
        previous_slice(ax)
        previous_slice(ab)
    elif event.key == 'k' or event.key == 'right':
        next_slice(ax)
        next_slice(ab)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
#########
# Statistics
#########
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h, h

####################
#
#   Augmentation
#
####################
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))#, np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))#, np.reshape(z, (-1, 1))

    distored_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def elastic_3d_transform(d3_img, verbose=False, seed=1):
    random.seed(seed)
    factor_row = random.uniform(0.2,0.8)
    factor_col = random.uniform(0.01, 0.08)
    #print(factor_row,factor_col)
    d3_img_tmp = d3_img
    img_rows = d3_img_tmp.shape[1]
    img_cols =  d3_img_tmp.shape[2]
    for img in d3_img_tmp:
        img = elastic_transform(img, img_rows*factor_row, sigma= img_cols*factor_col, random_state=None)
        if verbose:
            plt.imshow(img)
            plt.show()
    return d3_img_tmp

def ApplyAugmentation(d3_img, type_of_augmentation=None, dict_parameter=None, seed=1):
    random.seed(seed)
    patch_size = d3_img.shape
    d3_img_ = d3_img.reshape((patch_size[0], patch_size[1], patch_size[2]))
    if dict_parameter is None:
        dict_parameter={'rotation_xy':[-10,10],
                        'rotation_zx' :[-10,10],
                        'rotation_zy' :[-10,10],
                        'zooming':[1.01,1.1],
                        'down_scale':[0.85,0.99]
                        }
    if type_of_augmentation is None:
        seq=['None',
            'rotation_xy',
            #'rotation_zx',
            'rotation_zy',
            #'zooming',
            'h_flip',
            #'elastic'
            'v_flip',
            'z_flip'
            'rotate_90_k1',
            'h_flip',
            #'v_flip',
            #'z_flip',
            'rotate_90_k1',
            'rotate_90_k2',
            'rotate_90_k3'
            ]
        type_of_augmentation = random.choice(seq)

    if type_of_augmentation=='rotation_xy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img_, angle, axes=(1,2),reshape=False)
    elif type_of_augmentation=='rotation_zx':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img_, angle, axes=(0,2),reshape=False)
    elif type_of_augmentation=='rotation_zy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img_, angle, axes=(0,1),reshape=False)
    elif type_of_augmentation=='zooming':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img_, (1, value_factor, value_factor))
        x_a = new_img_zoom.shape[2]//2 - (patch_size[1]//2)
        y_a = new_img_zoom.shape[1]//2 - (patch_size[2]//2)
        new_3d_img = new_img_zoom[:,y_a:y_a+patch_size[1], x_a:x_a+patch_size[2]]
    elif type_of_augmentation=='down_scale':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img_, (1, value_factor, value_factor))
        new_img_zoom_tmp = np.zeros_like(d3_img_)
        x_a = new_img_zoom.shape[2]//2
        y_a = new_img_zoom.shape[1]//2
        x_a_b = new_img_zoom_tmp.shape[2]//2 - x_a
        y_a_b = new_img_zoom_tmp.shape[1]//2 - y_a
        new_img_zoom_tmp[:,y_a_b:y_a_b+new_img_zoom.shape[1], x_a_b:x_a_b+new_img_zoom.shape[1]] = new_img_zoom
        new_3d_img = new_img_zoom_tmp.copy()
    elif type_of_augmentation == 'h_flip':
        new_3d_img = np.flip(d3_img_,axis=1)
    elif type_of_augmentation == 'v_flip':
        new_3d_img = np.flip(d3_img_,axis=2)
    elif type_of_augmentation == 'z_flip':
        new_3d_img = np.flip(d3_img_,axis=0)
    elif type_of_augmentation=='rotate_90_k1':
        new_3d_img = np.rot90(d3_img_,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k2':
        new_3d_img = np.rot90(d3_img_,k=2,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k3':
        new_3d_img = np.rot90(d3_img_,k=3,axes=(1,2))
        '''
        elif type_of_augmentation=='elastic':
            transformation = augment.create_identity_transformation(d3_img.shape)
            # jitter in 3D
            transformation += augment.create_elastic_transformation(
                d3_img.shape,
                control_point_spacing=100,
                jitter_sigma=0.2)
            # apply transformation
            new_3d_img = augment.apply_transformation(d3_img, transformation)
        '''
    else:
        new_3d_img = d3_img_
    new_3d_img[new_3d_img>1.] = 1.
    new_3d_img[new_3d_img<0.] = 0. 
    '''
    bool_val = random.choice(['T', 'F'])
    if bool_val=='T':
        new_3d_img = elastic_3d_transform(new_3d_img, seed=seed)
    '''
    return new_3d_img.reshape(patch_size)

##########
#
#   Preprocessing
#
###########
def SharpTheImage(img, verbose=False, type_of_sharpness="Classic"):
    if type_of_sharpness == 'Classic':
        gaussian_1 = cv2.GaussianBlur(img, (9,9),10.0) #(9,9), 10.0)
        if verbose:
            print('gaussian_1')
            plt.imshow((gaussian_1))
            plt.show()
        img_copy = cv2.addWeighted(img, 1.5, gaussian_1, -0.5, 0, img) 
        if verbose:
            print('img_copy')
            plt.imshow((img_copy))
            plt.show()
        return img_copy
    elif type_of_sharpness == 'Curvature':
        imgOriginal = sitk.GetImageFromArray(img)
        imgSmooth = sitk.CurvatureFlow(image1=imgOriginal,
                                       timeStep=0.2,
                                       numberOfIterations=5)
        return sitk.GetArrayFromImage(imgSmooth)
    elif type_of_sharpness == 'CV2':
        blurred1 = cv2.fastNlMeansDenoising(np.uint8(img * 255), None, 41, 5, 17)
        return np.float32(blurred1/255)
    elif type_of_sharpness == 'Andrew1':
        gaussian_1 = cv2.GaussianBlur(img, (9,9),1.0)
        return gaussian_1
    elif type_of_sharpness == 'Andrew2':
        gaussian_1 = cv2.GaussianBlur(img, (9, 9), 15.0)
        return gaussian_1
    elif type_of_sharpness == 'Andrew3':
        gaussian_1 = cv2.GaussianBlur(img, (9, 9), 225.0)
        return gaussian_1
    elif type_of_sharpness == 'Bilateral':
        blur1 = cv2.bilateralFilter(np.float32(img),5,7,7)
        return blur1
    elif type_of_sharpness == 'TwoLevel':
        from scipy import ndimage
        blurred_f = ndimage.gaussian_filter(img, 2)
        if verbose:
            plt.imshow((blurred_f))
            plt.show()
                
        filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        if verbose:
            print('filter_blurred_f')
            plt.imshow((filter_blurred_f))
            plt.show()
        alpha = 30
        sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
        if verbose:
            print('sharpened')
            plt.imshow((sharpened))
            plt.show()
    elif type_of_sharpness == 'EDGE_ENHANCE':
        kernel = np.array(([-1, -1, -1],[-1, 15, -1],[-1, -1, -1]), dtype='int')
        sharpened = cv2.filter2D(img, -1, kernel)
        if verbose:
            plt.imshow(sharpened)
            plt.show()

    elif type_of_sharpness== 'Convolute':
        kernel = np.array((
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]), dtype="int")
        #convoleOutput = convolve(gray, kernel)
        sharpened = cv2.filter2D(img, -1, kernel)
        if verbose:
            plt.imshow(sharpened)
            plt.show()
    else:
        sharpened = img
    
    return sharpened
def Sharp3DVolume(d3_img, verbose=False, type_of_sharpness="Classic"):
    d3_img_tmp = d3_img.copy()
    for index in range(d3_img.shape[0]):
        d3_img_tmp[index] = SharpTheImage(d3_img_tmp[index], verbose, type_of_sharpness)
    return d3_img_tmp

def D3_equalize_adapthist(d3_img, clip_limit=0.05, nbins=1000):
    d3_img_tmp = np.zeros_like(d3_img).astype(np.float)
    for index in range(d3_img_tmp.shape[0]):
        img = d3_img[index].copy()
        imgs_x = exposure.equalize_adapthist(img, clip_limit=clip_limit, nbins=nbins)
        d3_img_tmp[index] = imgs_x
    return d3_img_tmp

def ReadImages(path='./', apply_sort=True):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    if apply_sort:
        dicom_names = SortDicomFiles(dicom_names)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def SortDicomFiles(files):
    files_lst = {}
    for filename in files:
        mri = sitk.ReadImage(filename)
        files_lst[int(mri.GetMetaData("0020|0013"))] = filename
    files_sorted = []
    for key in sorted(files_lst):
        files_sorted.append(files_lst[key])
    return files_sorted
#################################################
#
#               LOAD AND GENERATE MRI AND SEG DATA
#
################################################
def LoadFile(filename='', mask_filename=None, standard_volume=[24,384,384], normalize='CLAHE', verbose=False, apply_curve_smoothing=False):
    img = sitk.ReadImage(filename)
    #Normalize
    ct_scan = sitk.GetArrayFromImage(img)
    ct_scan = ct_scan.astype(int)
    if normalize=='CLAHE':
        ct_scan = D3_equalize_adapthist(ct_scan, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
    elif normalize=='CLAHE2':
        ct_scan /= np.max(ct_scan)
        ct_scan = D3_equalize_adapthist(ct_scan, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
    elif normalize=='ZSCORE':
        ct_scan = (ct_scan - np.mean(ct_scan))/(np.std(ct_scan))
    elif normalize=='MAX':
        ct_scan = ct_scan / np.max(ct_scan)
    else:
        ct_scan = ct_scan
    
    #Sharpen
    ct_scan = Sharp3DVolume(ct_scan, verbose, type_of_sharpness=configuration.type_of_sharpness)



    #Convert to sitk format
    img = sitk.GetImageFromArray(ct_scan) 
    

    if verbose:
        print('img', img.GetDimension(), img.GetDirection(), img.GetOrigin(), img.GetSpacing())
    if mask_filename is not None:
        mask = sitk.ReadImage(mask_filename)
        mask_ = sitk.GetArrayFromImage(mask)
        mask_ = GenerateMask(mask_, verbose)
        mask = sitk.GetImageFromArray(mask_)
        mask.SetDirection(img.GetDirection())
        mask.SetOrigin(img.GetOrigin())
        mask.SetSpacing(img.GetSpacing())
        if verbose:
            print('mask',mask.GetDimension(), mask.GetDirection(), mask.GetOrigin(), mask.GetSpacing())
    else:
        print('Warn: No mask file was given...')
    #Resize Standard
    if verbose:
        mask_tmp = sitk.GetArrayFromImage(mask)
        img_tmp = sitk.GetArrayFromImage(img)
        img_tmp = img_tmp/np.max(img_tmp)
        multi_slice_viewer(img_tmp,mask_tmp, 1)
        plt.show()
    new_x_size = standard_volume[1]
    new_y_size = standard_volume[2]
    new_z_size = standard_volume[0]
    # Create the reference image with a zero origin, identity direction cosine matrix and dimension   
    new_size = [new_x_size, new_y_size, new_z_size]
    new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), new_size)]
    
    interpolator_type = sitk.sitkLanczosWindowedSinc
    new_img = sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, img.GetPixelIDValue())
    
    if mask_filename is not None:
        interpolator_type = sitk.sitkNearestNeighbor
        mask = sitk.Resample(mask, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, mask.GetPixelIDValue())

    if verbose:
        mask_tmp = sitk.GetArrayFromImage(mask)
        img_tmp = sitk.GetArrayFromImage(new_img)
        img_tmp = img_tmp/np.max(img_tmp)
        multi_slice_viewer(mask_tmp,img_tmp)
        plt.show()

    if apply_curve_smoothing:
        ct_scan = smooth_images(ct_scan)
        print('apply_curve_smoothing')
        if verbose:
            multi_slice_viewer(ct_scan,mask_tmp, 1)
            plt.show()

    ct_scan = sitk.GetArrayFromImage(new_img)
    ct_scan[ct_scan>1] = 1.
    ct_scan[ct_scan<0] = 0.

    #Convert Segment data to Numpy
    if mask_filename is not None:
        mask = sitk.GetArrayFromImage(mask)
        return ct_scan, mask
    else:
        return ct_scan

def GenerateMask(ct_mask, verbose):
    #Label contour
    ct_mask_data = ct_mask.copy()
    for n in range(ct_mask.shape[0]):    
        label_boolean = ct_mask_data[n]
        label_boolean = label_boolean.astype(np.uint8)
        label_boolean = label_boolean *255
        ret, thresh = cv2.threshold(label_boolean,127,255,0)
        if verbose:
            plt.imshow(thresh)
            plt.show()
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        points_list = []
        for contour in contours:
            points = list(contour)
            points.append(points[0])
            points = np.array(points)
            points = points.astype(int)
            points_list.append(points)
        for point_lst in points_list:
            cv2.polylines(ct_mask_data[n],[point_lst],True,1,3)
    return ct_mask_data
        
def GenerateDataSet(filenames, mask_filenames, verbose=False):
    data_shape = (len(filenames), configuration.standard_volume[0], configuration.standard_volume[1],configuration.standard_volume[2],configuration.D3_channel)
    if configuration.output_channel==0:
        segment_shape = (len(filenames), configuration.standard_volume[0], configuration.standard_volume[1],configuration.standard_volume[2])
    else:
        segment_shape = (len(filenames), configuration.standard_volume[0], configuration.standard_volume[1],configuration.standard_volume[2], configuration.output_channel)

    #Generate on mem
    data_ = np.zeros(data_shape)
    mask_ = np.zeros(segment_shape)
    #Store into the mem.
    counter_ = 0
    print('Proc: Loading and Storing the MRI')
    for filename, mask_filename in zip(filenames, mask_filenames):
        img, mask = LoadFile(filename, mask_filename,standard_volume=configuration.standard_volume, normalize=configuration.normalize, verbose=configuration.verbose)
        if verbose:
            multi_slice_viewer(mask,img)
            plt.show()
        if len(data_shape)==5:
            data_[counter_] = img.reshape(data_shape[1],data_shape[2],data_shape[3],data_shape[4])
        else:
            data_[counter_] = img.reshape(data_shape[1],data_shape[2],data_shape[3])
        mask_[counter_] = mask
        counter_ += 1 
        print('Done:',filename, '|', mask_filename)
    #Save
    print('Done: Loading and Storing the MRI')
    return data_, mask_

def GenerateDataSetSaveResults(path=None,ratio=0.7):
    #Generate numpy list
    if path is None:
        print('ERR: Please provide the path...')
        return

    print('path:',path)
    onlyfiles = [f for f in listdir(path) if (isfile(join(path, f)) and (os.path.splitext(f)[1] in [".mhd"]))]
    
    def last_6chars(x):
        return(x[0:6])

    onlyfiles = sorted(onlyfiles, key = last_6chars) 
    segmentation_file_list =[]
    image_file_list =[]
    for file in onlyfiles:
        print(file)
        if ('segmentation' in file):
            segmentation_file_list.append(file)
        else:
            image_file_list.append(file)
    
    from random import shuffle
    x = [i for i in range(len(segmentation_file_list))]
    shuffle(x)
    
    image_file_list_ = []
    for index in x:
        image_file_list_.append(join(path, image_file_list[index]))

    image_file_list = image_file_list_

    segmentation_file_list_ = []
    for index in x:
        segmentation_file_list_.append(join(path, segmentation_file_list[index]))
    

    segmentation_file_list = segmentation_file_list_

    nr_train = int(round(ratio*len(segmentation_file_list)))
    
    img_train_set = image_file_list[0:nr_train]
    mask_train_set = segmentation_file_list[0:nr_train]
    
    img_valid_set = image_file_list[nr_train:len(segmentation_file_list)]
    mask_valid_set = segmentation_file_list[nr_train:len(segmentation_file_list)]
    info_data = {'train': (img_train_set, mask_train_set), 'valid' : (img_valid_set,mask_valid_set)}

    for val in info_data.keys():
        data_img, data_mask = info_data[val]
        mri_slices, mask_slices = GenerateDataSet(data_img,data_mask)
        print(mri_slices.shape)
        print(mask_slices.shape)
        print("Proc: Save the result...")
        np.save('./img_%s_set.npy' % (val), mri_slices)
        np.save('./mask_%s_set.npy' % (val), mask_slices)
        print('Done: Saving the result...')
 
###########
# Learning rate optimizer
##########
import math
def step_decay(epoch, lr=1e-4):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 2.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

###########
#
#   DataGenerator
#
###########
def Normalize(data):
    X_train_pos_ = data
    vle_min = np.min(X_train_pos_)
    vle_max = np.max(X_train_pos_)
    for index in range(X_train_pos_.shape[0]):
        X_train_pos = X_train_pos_[index]
        mean_ = np.mean(X_train_pos)
        std_ = np.std(X_train_pos)
        data_ = np.tanh((X_train_pos-mean_)/std_) 
        X_train_pos_[index] = data_
    return X_train_pos_

def DataGeneratorNative(X,Y, batch_size=12, RunNormalize=True):
    steps=X.shape[0]//batch_size
    X_ = X#[0]
    #X_1 = X[1]
    while 1:
        for i in range(0,steps):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            if start_index>(X_.shape[0]-batch_size):
                start_index = X_.shape[0]-batch_size
                end_index = X_.shape[0]
            
            x_batch_tmp_0 = X_[start_index:end_index].copy()
            if RunNormalize:
                for index in range(x_batch_tmp_0.shape[0]):
                    x_batch_tmp_0[index] = Normalize(x_batch_tmp_0[index])
            y_batch_tmp = Y[start_index:end_index]
            
            yield x_batch_tmp_0, y_batch_tmp#[x_batch_tmp_0,x_batch_tmp_1], y_batch_tmp

def DataGenerator_OneShot_Learning_X_Y(X_data, negative_lst, positive_lst):
    print('Generating random set...')
    X_train_p = X_data[positive_lst]
    Y_neg = negative_lst[:len(positive_lst)]
    X_train_n = X_data[negative_lst]
    Y_different = np.zeros((X_train_p.shape[0]))
    Y_ones_p = np.ones((X_train_p.shape[0]))
    Y_ones_n = np.ones((X_train_n.shape[0]))
    negative_lst_ = negative_lst.copy()
    positive_lst_ = positive_lst.copy()
    random.shuffle(negative_lst_)
    random.shuffle(positive_lst_)


    Y_all = np.concatenate([Y_ones_p,Y_ones_p,Y_different,Y_different, Y_ones_n, Y_ones_n])
    X_all_0= np.concatenate([X_train_p, X_data[positive_lst_], X_train_p,X_data[Y_neg], X_data[negative_lst_], X_train_n])
    X_all_1= np.concatenate([X_train_p, X_train_p, X_data[Y_neg],X_train_p, X_train_n, X_train_n])
    for i in range(4):
        random.shuffle(negative_lst_)
        negative_lst_2 = negative_lst_.copy()
        random.shuffle(negative_lst_2)

        random.shuffle(positive_lst_)
        positive_lst_2 = positive_lst_.copy()
        random.shuffle(positive_lst_2)
        negative_lst_d = negative_lst_[:len(positive_lst_)]
        Y_all = np.concatenate([Y_all, Y_ones_p, Y_different,Y_ones_n]) 
        X_all_0= np.concatenate([X_all_0, X_data[positive_lst_],X_data[positive_lst_], X_data[negative_lst_]])
        X_all_1= np.concatenate([X_all_1, X_data[positive_lst_2],X_data[negative_lst_d], X_data[negative_lst_2]])

    index_range = list(range(0,Y_all.shape[0]))
    np.random.shuffle(index_range)
    Y_all = Y_all[index_range]
    X_all_0 = X_all_0[index_range]
    X_all_1 = X_all_1[index_range]
    return (X_all_0,X_all_1), Y_all#%%

def Generate_positive_negative_lists(Y, Positive_Value=True, Negative_Value=False):
    Y_neg = list(np.where(Y==Negative_Value)[0].flatten())
    Y_pos = list(np.where(Y==Positive_Value)[0].flatten())
    random.shuffle(Y_neg)
    random.shuffle(Y_pos)
    return Y_pos, Y_neg

def DataGeneratorWithAugmentation(X,Y, batch_size=12, RunNormalize=True, RunAugmentation=False, HED=False, max_iteration_per_epoch=10000):
    steps= X.shape[0]//batch_size
    seed_counter  = 0
    indexes = list(range(X.shape[0]))
    random.shuffle(indexes)
    X_ = X[indexes]
    Y_ = Y[indexes]
    shape_p = (batch_size,configuration.standard_volume[0],configuration.standard_volume[1],configuration.standard_volume[2],1)
    while 1:
        if seed_counter >= max_iteration_per_epoch:
            seed_counter = 0
        for i in range(0,steps):
            random.seed(seed_counter)
            selected_index_patch = random.sample(list(range(X_.shape[0])), batch_size)
            x_batch_tmp = X_[selected_index_patch].copy()
            y_batch_tmp = Y_[selected_index_patch]
            seed_counter += 1
            for index in range(x_batch_tmp.shape[0]):
                if RunNormalize:
                    x_batch_tmp[index] = Normalize(x_batch_tmp[index])
                if RunAugmentation:
                    x_batch_tmp[index] = ApplyAugmentation(x_batch_tmp[index], seed=seed_counter)
            if HED:
                one_batch_y =y_batch_tmp.reshape(shape_p)
                
                yield x_batch_tmp, [one_batch_y]*7
            else:
                yield x_batch_tmp, y_batch_tmp.reshape(shape_p)

def GenerateEqualPositiveAndNegativeValue(X,Y, batch_size=12, RunNormalize=True, max_iteration_per_epoch=10000):
    P_indexes, N_indexes = Generate_positive_negative_lists(Y)
    random.shuffle(N_indexes)
    random.shuffle(P_indexes)
    steps=X.shape[0]//batch_size
    counter  = 0
    while 1:
        if counter>=max_iteration_per_epoch:
            counter =0
        
        for i in range(0,steps):
            random.seed(
                counter
            )
            N_indexes_randomly_selected = random.sample(N_indexes, batch_size//2)
            P_indexes_randomly_selected = random.sample(P_indexes, batch_size//2)
            list_selectred = N_indexes_randomly_selected + P_indexes_randomly_selected
            random.shuffle(list_selectred)
            x_batch_tmp = X[list_selectred].copy()
            y_batch_tmp = Y[list_selectred]
            counter += 1 
            if RunNormalize:
                for index in range(x_batch_tmp.shape[0]):
                    x_batch_tmp[index] = Normalize(x_batch_tmp[index])
                    x_batch_tmp[index] = ApplyAugmentation(x_batch_tmp[index], seed=counter)
            yield x_batch_tmp, y_batch_tmp

########
#
#   Evaluation
#
########
from enum import Enum

# Use enumerations to represent the various evaluation measures
def EvaluteTheSequence(filename, mask_filename, segmentations, verbose=False):
    image, mask = LoadFile(filename, mask_filename, configuration.standard_volume, configuration.normalize)
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)
    height = image.GetHeight()
    width = image.GetWidth()
    print('segmentations.shape', segmentations.shape)
    seg = sitk.GetImageFromArray(segmentations)
    seg.SetDirection(image.GetDirection())
    seg.SetOrigin(image.GetOrigin())
    seg.SetSpacing(image.GetSpacing())
    Evaluate(mask, seg)

class OverlapMeasures(Enum):
    jaccard, dice, volume_similarity, false_negative, false_positive = range(5)

class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance = range(5)
results_x = {}
for parameter in ['Dice', 'Jaccard', 
                  'Similarity', 'GetFalseNegativeError', 
                  'GetFalsePositiveError','Hausdorff', 
                  'mean_surface_distance','median_surface_distance',
                  'std_surface_distance','max_surface_distance'
                   ]:
    results_x[parameter] = []

def Evaluate(reference_segmentation_f, segmentations_f):
    # Empty numpy arrays to hold the results 
    reference_segmentation = reference_segmentation_f #sitk.GetImageFromArray(reference_segmentation_f, isVector=False)
    segmentations = segmentations_f# sitk.GetImageFromArray(segmentations_f,isVector=False)
    segmentations = sitk.BinaryThreshold(segmentations, lowerThreshold=0.5, upperThreshold=1)
    reference_segmentation = sitk.BinaryThreshold(reference_segmentation, lowerThreshold=1, upperThreshold=1)
    
    overlap_results = np.zeros((len(segmentations),len(OverlapMeasures.__members__.items())))  
    surface_distance_results = np.zeros((len(segmentations),len(SurfaceDistanceMeasures.__members__.items())))

    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False))
    reference_surface = sitk.LabelContour(reference_segmentation)

    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    #for i, seg in enumerate(segmentations):
    # Overlap measures
    seg = segmentations
    i = 0
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(reference_segmentation, seg)
    results_x['Dice'].append(overlap_measures_filter.GetDiceCoefficient())
    results_x['Jaccard'].append(overlap_measures_filter.GetJaccardCoefficient())
    results_x['Similarity'].append(overlap_measures_filter.GetVolumeSimilarity())
    results_x['GetFalseNegativeError'].append(overlap_measures_filter.GetFalseNegativeError())
    results_x['GetFalsePositiveError'].append(overlap_measures_filter.GetFalsePositiveError())
    
    print('Dice',overlap_measures_filter.GetDiceCoefficient())
    print('Jaccard',overlap_measures_filter.GetJaccardCoefficient())
    print('Similarity',overlap_measures_filter.GetVolumeSimilarity())
    print('GetFalseNegativeError',overlap_measures_filter.GetFalseNegativeError())
    print('GetFalsePositiveError',overlap_measures_filter.GetFalsePositiveError())
    
    overlap_results[i,OverlapMeasures.jaccard.value] = overlap_measures_filter.GetJaccardCoefficient()
    overlap_results[i,OverlapMeasures.dice.value] = overlap_measures_filter.GetDiceCoefficient()
    overlap_results[i,OverlapMeasures.volume_similarity.value] = overlap_measures_filter.GetVolumeSimilarity()
    overlap_results[i,OverlapMeasures.false_negative.value] = overlap_measures_filter.GetFalseNegativeError()
    overlap_results[i,OverlapMeasures.false_positive.value] = overlap_measures_filter.GetFalsePositiveError()
    # Hausdorff distance
    hausdorff_distance_filter.Execute(reference_segmentation, seg)
    surface_distance_results[i,SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    print('Hausdorff', hausdorff_distance_filter.GetHausdorffDistance())
    results_x['Hausdorff'].append(hausdorff_distance_filter.GetHausdorffDistance())
    
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False))
    segmented_surface = sitk.LabelContour(seg)
            
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
            
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
        
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                            list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                            list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
            
    all_surface_distances = seg2ref_distances + ref2seg_distances
        
    surface_distance_results[i,SurfaceDistanceMeasures.mean_surface_distance.value] = np.mean(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.median_surface_distance.value] = np.median(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    surface_distance_results[i,SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)
    print('mean_surface_distance', np.mean(all_surface_distances))
    print('median_surface_distance', np.median(all_surface_distances))
    print('std_surface_distance', np.std(all_surface_distances))
    print('max_surface_distance', np.max(all_surface_distances))
    results_x['mean_surface_distance'].append(np.mean(all_surface_distances))
    results_x['median_surface_distance'].append(np.median(all_surface_distances))
    results_x['std_surface_distance'].append(np.std(all_surface_distances))
    results_x['max_surface_distance'].append(np.max(all_surface_distances))


    # Print the matrices
    np.set_printoptions(precision=3)

        # Graft our results matrix into pandas data frames 
    overlap_results_df = pd.DataFrame(data=overlap_results, index = list(range(len(segmentations))), 
                                    columns=[name for name, _ in OverlapMeasures.__members__.items()]) 
    surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(len(segmentations))), 
                                    columns=[name for name, _ in SurfaceDistanceMeasures.__members__.items()])

    #print(overlap_results_df)
    #print(surface_distance_results_df) 
from skimage.morphology import remove_small_objects
def SegmentProstateSecondStep(mri_seq, model, verbose=False, ShowResult=True):
    #Prepare
    npy_mri_seq = LoadFile(mri_seq, standard_volume=configuration.standard_volume)
    input_shape= (1,configuration.standard_volume[0],configuration.standard_volume[1], configuration.standard_volume[2], configuration.D3_channel)
    npy_mri_seq_ = npy_mri_seq.reshape(input_shape)
    
    #Predict
    heatmap_predicted_y_x = model.predict(npy_mri_seq_)
    prostate_segmentation = heatmap_predicted_y_x[0]
    for i, img in enumerate(prostate_segmentation):
        img = img>0.5
        #Remove small objects
        improved_seg = remove_small_objects(img, min_size=300, connectivity=8, in_place=False)
        prostate_segmentation[i] = improved_seg
    
    reference = []
    for i in range(prostate_segmentation.shape[0]):
        x = np.count_nonzero(prostate_segmentation[i])
        heatmap_predicted_y_x = prostate_segmentation[i]
        #Document the continuation of the prostate segmentation
        print(x)
        if x >=300:
            reference.append(1)
        else:
            heatmap_predicted_y_x=0.
            reference.append(0)

        #Add into the 3D model
        prostate_segmentation[i,:,:] = heatmap_predicted_y_x


    
    #Check for the continuation of the prostaste segmentation.
    length_segm =len(reference)
    print('length_segm',length_segm)
    print(reference)
    count=0
    stop_index=-1
    start_index=-1
    for index in range(length_segm):
        if index<length_segm-1:
            previous_index = index -1
            next_index = index +1
            if reference[previous_index] ==0 and reference[index] == 1 and reference[next_index] == 0:
                reference[index]= 0
            elif reference[previous_index] ==0 and reference[index] == 1:
                start_index = index
            
            if reference[next_index] ==0 and reference[index] == 1:
                diff = index - start_index
                if diff>4:
                    stop_index = next_index
                    break
        
    if stop_index>0:
        for index in range(stop_index,length_segm):
            reference[index] = 0.

    print('after correction:',reference)
    #Multiply by the continuation factor to remove object outside the continuation of the prostate segmentation..
    for index in range(prostate_segmentation.shape[0]):
        prostate_segmentation[index] = reference[index] * prostate_segmentation[index]
    
    #Visualize the result
    if ShowResult or verbose:
        multi_slice_viewer(prostate_segmentation, npy_mri_seq, depth=1)
        plt.show()
    return prostate_segmentation

def SegmentProstate(dicom, model):
    seg = []
    directory_mode = configuration.directory_mode
    test_mode = configuration.test_mode
    verbose =configuration.verbose_Test
    ShowResult = configuration.ShowResult
    if directory_mode:
        print(dicom)
        fileList = listdir(dicom)
        fileList = filter(lambda x: '.mhd' in x, fileList)

        def last_6chars(x):
            return(x[0:6])

        onlyfiles = sorted(fileList, key = last_6chars) 
        segmentation_file_list =[]
        image_file_list =[]
        for file in onlyfiles:
            print(file)
            if ('segmentation' in file):
                segmentation_file_list.append(file)
            else:
                image_file_list.append(file)

        print(image_file_list)
        for index, filename in enumerate(image_file_list):
            mri_seq = sitk.ReadImage(dicom+filename)
            #mri_seq = smooth_images(mri_seq)
            seg = SegmentProstateSecondStep(mri_seq, model, verbose, ShowResult)
            if test_mode:
                EvaluteTheSequence(dicom+filename, dicom+segmentation_file_list[index],seg)
        for key in results_x.keys():
            mean_x = np.mean(np.array(results_x[key]))
            median_x = np.median(np.array(results_x[key]))
            max_x = np.max(np.array(results_x[key]))
            min_x = np.min(np.array(results_x[key]))
            std_x = np.std(np.array(results_x[key]))
            m_CI = mean_confidence_interval(results_x[key])
            print(key,'mean: ', mean_x,'std',std_x,'median',median_x, 'min',min_x, 'max', max_x)
            print(key, 'mean, 95%CI:', m_CI)
    else:
        mri_seq =ReadImages(path=dicom)
        seg.append(SegmentProstateSecondStep(mri_seq, model, verbose, ShowResult))
    return seg

