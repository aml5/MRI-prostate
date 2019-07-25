import numpy as np
from matplotlib import pyplot as plt
import csv
import math
from skimage.exposure import rescale_intensity
from skimage.color import rgb2grey, separate_stains, rgb2hsv, rgb2lab
from numpy import linalg
import random
import cv2
import os
from functools import partial
import keras.backend as K
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])
    
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.
        This loss is useful when you have unbalanced numbers of pixels within an image
        because it gives all classes equal weight. However, it is not the defacto
        standard for image segmentation.
        For example, assume you are trying to predict if each pixel is cat, dog, or background.
        You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
        should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?
        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
        Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
        What is a good evaluation measure for semantic segmentation?.
        IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
        https://en.wikipedia.org/wiki/Jaccard_index
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
    
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
def bilinear_interpolate_two_images(img_1, img_2):
    img = img_1 + img_2
    img_b = bilinear_interpolate(img)
    av_img_b = img_b * 0.5
    return av_img_b

def bilinear_interpolate(im):
    y = range(im.shape[0])
    x = range(im.shape[1])
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # wa*Ia + wb*Ib + wc*Ic + wd*Id
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T
    
def EnhanceColor(img, clipLimit=20, tileGridSize=(20,20)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_c = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #plt.imshow(img_c)
    #plt.show()
    return img_c

def stainspace_to_2d_array(ihc_xyz, channel):
    rescale = rescale_intensity(ihc_xyz[:, :, channel], out_range=(0,1))
    stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
    grey_array = rgb2grey(stain_array)
    return grey_array

def Convert_to_OD(rgb_255, normalized=False):
    rgb_dec = (rgb_255+1.) / 256.
    #print(np.min(rgb_dec))
    OD = -1 * np.log10(rgb_dec)
    '''
    if normalized:
        p_squared_sum = (OD[:,:,0] ** 2) + (OD[:,:,1] ** 2) + (OD[:,:,1] ** 2)
        OD[:,:,0] = OD[:,:,0] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,1] = OD[:,:,1] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,2] = OD[:,:,2] / (np.sqrt(p_squared_sum)+1e-7)
    '''
    return OD
def Convert_to_HE_Normalized(rgb_255):
    img = Convert_to_HRD(rgb_255)
    img_hsv = Convert_to_HSV(rgb_255)
    img[:,:,0] = (img[:,:,0]-(-1.5132374498073249))/(0+1.5132374498073249)
    img[:,:,1] = img_hsv[:,:,1] # (img[:,:,1]-(0))/(0.6109545548214189+0)
    img[:,:,2] = img_hsv[:,:,2] # (img[:,:,2]-(-1.103678089452847))/(0+1.103678089452847) 
    #img[:,:,1] = (img_lab[:,:,1] + 128) /(128*2)
    #img[:,:,2] = (img_lab[:,:,2] + 128) /(127+128)
    #plt.imshow(img)
    #plt.show()
    return img

def Convert_to_HE_Normalized_2(rgb_255):
    img = Convert_to_HRD(rgb_255)
    #img_lab = Convert_to_LAB(rgb_255)
    img[:,:,0] = (img[:,:,0]-(-1.5132374498073249))/(0+1.5132374498073249)
    img[:,:,1] =  (img[:,:,1]-(0))/(0.6109545548214189+0)
    img[:,:,2] =  (img[:,:,2]-(-1.103678089452847))/(0+1.103678089452847) 
    #img[:,:,1] = (img_lab[:,:,1] + 128) /(128*2)
    #img[:,:,2] = (img_lab[:,:,2] + 128) /(127+128)
    return img

def Convert_to_HSV(rgb_255, normalized=False):
    img = rgb2hsv(rgb_255)
    #rgb_dec = (img+1.) / 256.
    #print(np.min(rgb_dec))
    #OD = -1 * np.log10(img)
    '''
    if normalized:
        p_squared_sum = (OD[:,:,0] ** 2) + (OD[:,:,1] ** 2) + (OD[:,:,1] ** 2)
        OD[:,:,0] = OD[:,:,0] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,1] = OD[:,:,1] / (np.sqrt(p_squared_sum)+1e-7)
        OD[:,:,2] = OD[:,:,2] / (np.sqrt(p_squared_sum)+1e-7)
    '''
    #plt.imshow(OD)
    #plt.show()
    return img

def Convert_to_LAB(rgb_255):
    img = rgb2lab(rgb_255)
    return img


def LN_normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def Convert_to_HRD(image):
    from skimage.color import rgb2hed
    ''''
    img_new = np.zeros(image.shape, dtype='float')
    rgb_from_hrd = np.array([[0.644, 0.710, 0.285],
                    [0.0326, 0.873, 0.487],
                    [0.270, 0.562, 0.781]])

    #conv_matrix
    hrd_from_rgb = linalg.inv(rgb_from_hrd)
    #Seperate stain
    ihc_hrd = separate_stains(image, hrd_from_rgb)

    #img_new = LN_normalize(ihc_hrd)
    #Stain space conversion
    DAB_Grey_Array = stainspace_to_2d_array(ihc_hrd, 2)
    Hema_Gray_Array = stainspace_to_2d_array(ihc_hrd, 0)
    GBIred_Gray_Array = stainspace_to_2d_array(ihc_hrd, 1)
    img_new = np.stack((Hema_Gray_Array, DAB_Grey_Array,GBIred_Gray_Array), axis=-1)
    '''
    ihc_hed = rgb2hed(image)
    #stainspace_to_2d_array()
    return ihc_hed

def stainspace_to_2d_array(ihc_xyz, channel):
    rescale = rescale_intensity(ihc_xyz[:, :, channel], out_range=(-1,1))
    stain_array = np.dstack((np.zeros_like(rescale), rescale, rescale))
    grey_array = rgb2grey(stain_array)
    return grey_array

def normalize_mapping(source_array, source_mean, target_mean):
    for x in source_array:
        x[...] = x/source_mean
        x[...] = x * target_mean
    return source_array

def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def image_generator(generator, test=False, channel_number=1, reshape=False, HED=False, set_class=False,DualMode=False):
    while 1:
        if test:
            x_batch = generator.next()
        else:
            x_batch, y_batch = generator.next()
        x_batch_ = np.zeros((x_batch.shape[0], x_batch.shape[1],x_batch.shape[2],channel_number))
        if channel_number==3:
            for indx, img in enumerate(x_batch):
                img_tmp = np.zeros((img.shape[0],img.shape[1],channel_number))
                for i in range(3):
                    img_tmp[...,i] = img[:,:,0]
                x_batch_[indx] = img_tmp
            x_batch = x_batch_
        if set_class:
            y_class = np.zeros((x_batch.shape[0],2))
            for i, img_y in enumerate(y_batch):
                n_class = 0
                if np.count_nonzero(img_y[...,1])>0:
                    n_class = 1
                y_class[i, n_class]= 1.
            #print(y_class)
        if HED:
            x_batch, y_batch = generator.next()
            y_batch_tmp = np.zeros((y_batch.shape[0],y_batch.shape[1],y_batch.shape[2],1))
            y_batch_tmp[y_batch[...,1] > 0,1] = .1
            #y_batch_tmp[y_batch[...,2] > 0,1] = .1
            yield(x_batch,[y_batch,y_batch,y_batch,y_batch,y_batch,y_batch] )
        elif DualMode and test==False:
            contour_batch = np.zeros((y_batch.shape[0], y_batch.shape[1], y_batch.shape[2],1), dtype=K.floatx())
            contour_batch[..., 0] = y_batch[..., 2]
            area_batch = np.zeros((y_batch.shape[0], y_batch.shape[1], y_batch.shape[2],1), dtype=K.floatx())
            area_batch[..., 0] = y_batch[..., 1]
            if reshape:
                y_batch_tmp = np.zeros((y_batch.shape[0],y_batch.shape[1],y_batch.shape[2],2))
                y_batch_tmp[y_batch[...,0] > 0,0] = 1.
                y_batch_tmp[y_batch[...,1] > 0,1] = .1
                y_batch_tmp[y_batch[...,2] > 0,1] = .1
                y_batch = y_batch_tmp
            yield(x_batch,[y_batch, area_batch, contour_batch])
        elif reshape and test==False:
            y_batch_tmp = np.zeros((y_batch.shape[0],y_batch.shape[1],y_batch.shape[2],2))
            y_batch_tmp[y_batch[...,1] > 0,1] = .1
            y_batch_tmp[y_batch[...,2] > 0,1] = .1
            y_batch = y_batch_tmp#.reshape((y_batch.shape[0],y_batch.shape[1],y_batch.shape[2],1))
        elif test:
            yield(x_batch)
        elif set_class:
            yield(x_batch,[y_batch, y_class])
        else:
            yield(x_batch,y_batch)#, area_batch, contour_batch])

def random_scale_img(img, mask, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img, mask


    if not isinstance(img, mask, list):
        img = [img]
        mask = [mask]

    import cv2
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy:
        scale_y = scale_x

    org_height, org_width = img[0].shape[:2]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y

    res_img = []
    for img_inst in img:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
        res_img.append(tmp)

    res_mask = []
    for img_inst in mask:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(
            img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(
                scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(
                scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y +
                         org_height, start_x: start_x + org_width]
        res_img.append(tmp)

    return res_img, res_mask
# Add others as needed
probs = {'keep': 0.1,'elastic': 0.9}
def apply_transform(image, mask, img_target_cols):
        #prob_value = np.random.uniform(0, 1)
        #if prob_value > probs['keep']:
            # You can add your own logic here.
        sigma = np.random.uniform(img_target_cols * 0.20, img_target_cols * 0.20)
        image = elastic_transform(image, img_target_cols, sigma)
        mask = elastic_transform(mask, img_target_cols, sigma)

        # Add other transforms here as needed. It will cycle through available transforms with give probs

        #mask = mask.astype('float32') / 255.
        return image, mask

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    image_d = image.copy()
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[0:2]

    for i in range(image.shape[2]):
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                            sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        image_d[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape)

    return image_d

def CropLayers(self, input_shape=(64, 64, 3), cropping_size=(64, 64), off_set=(0, 0)):
        seq_seq_left_right_top_bottom_itms = []
        for x in range(0, input_shape[0], cropping_size[0]):
            for y in range(0, input_shape[1], cropping_size[1]):
                seq_top_bottom = (
                    off_set[0]+x, (off_set[0]+x+cropping_size[0]))
                seq_left_right = (
                    off_set[1]+y, (off_set[1]+y+cropping_size[1]))
                if seq_top_bottom[1] > input_shape[0]:
                    diff = seq_top_bottom[1] - input_shape[0]
                    seq_top_bottom = (seq_top_bottom[0] - diff, input_shape[0])
                if seq_left_right[1] > input_shape[1]:
                    diff = seq_left_right[1] - input_shape[1]
                    seq_left_right = (seq_left_right[0] - diff, input_shape[1])

                cropped_slide = (
                    seq_top_bottom[0], seq_top_bottom[1], seq_left_right[0], seq_left_right[1])
                seq_seq_left_right_top_bottom_itms.append(cropped_slide)
        return seq_seq_left_right_top_bottom_itms

def SplitAnImagesInSmallPatches(img, patch_size=(96,96), off_set=(0, 0)):
    input_shape = img.shape
    for x in range(0, input_shape[0], patch_size[0]):
        for y in range(0, input_shape[1], patch_size[1]):
            seq_top_bottom = (
                off_set[0]+x, (off_set[0]+x+cropping_size[0]))
            seq_left_right = (
                    off_set[1]+y, (off_set[1]+y+cropping_size[1]))
            if seq_top_bottom[1] > input_shape[0]:
                diff = seq_top_bottom[1] - input_shape[0]
                seq_top_bottom = (seq_top_bottom[0] - diff, input_shape[0])
            if seq_left_right[1] > input_shape[1]:
                diff = seq_left_right[1] - input_shape[1]
                seq_left_right = (seq_left_right[0] - diff, input_shape[1])

            cropped_slide = (
                seq_top_bottom[0], seq_top_bottom[1], seq_left_right[0], seq_left_right[1])
            seq_seq_left_right_top_bottom_itms.append(cropped_slide)
        return seq_seq_left_right_top_bottom_itms

if __name__=="__main__":
    plot_log('result/log.csv')



