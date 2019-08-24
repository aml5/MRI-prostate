from keras import backend as K
from keras.utils import Sequence
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import random
import scipy
import cv2
from keras import layers
import keras
import SimpleITK as sitk
import random
import h5py
import numpy as np
import scipy
#from mayavi import mlab
from scipy import ndimage
import matplotlib.pyplot as plt
import keras.backend as K
from skimage.exposure import equalize_adapthist
from keras.engine.topology import Layer
from keras.constraints import min_max_norm
from skimage import exposure
from sklearn.metrics import auc,average_precision_score, precision_recall_curve,classification_report, f1_score, confusion_matrix,brier_score_loss
from sklearn.metrics import roc_auc_score,roc_curve ,fowlkes_mallows_score
 

import scipy
def cutoff_youdens(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plotROCCurveMultiCall(plt, true_label, prob_pos, title):
    fpr, tpr, _ = roc_curve(true_label, prob_pos)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='orange',
            lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return plt

class JunctionWeightLayer(Layer): # a junction layer
    def __init__(self,  **kwargs):
        self.func_junction = layers.add
        super(JunctionWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape describes the number of the junctions.
        '''
        assert isinstance(input_shape, list)
        self.W1 = self.add_weight(name='junction_weight_first_element', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0, max_value=1))
        self.W2 = self.add_weight(name='junction_weight_second_element', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0, max_value=1))
        super(JunctionWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        a = a * self.W1 
        b = b * self.W2
        v = self.func_junction([a,b])
        return v

class RotationThetaWeightLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(RotationThetaWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        
        self.W1 = self.add_weight(name='kernel_01', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        self.W2 = self.add_weight(name='kernel_02', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        
        super(RotationThetaWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x

        return K.cos(self.W1*90) * (-2) * K.exp(-(a**2+b**2)) + K.sin(self.W2*90) * (-2) * b * K.exp(-(a**2+b**2))
class RotationOneInputThetaWeightLayer(Layer): # a scaled layer
    def __init__(self, **kwargs):
        super(RotationOneInputThetaWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        #assert isinstance(input_shape, list)
        
        self.W1 = self.add_weight(name='kernel_01', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        self.W2 = self.add_weight(name='kernel_02', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0))
        
        super(RotationOneInputThetaWeightLayer, self).build(input_shape)

    def call(self, x):
        return K.cos(self.W1*90) * x  + K.sin(self.W2*90) * x

def Normalize(data, normalize="tanh"):
    ct_scan=data.copy()
    #print("shape", ct_scan.shape)
    if normalize=='CLAHE':
        ct_scan = D3_equalize_adapthist(ct_scan.astype(np.uint16), clip_limit=0.05, nbins=1000)
    elif normalize=='CLAHE2':
        ct_scan /= np.max(ct_scan)
        ct_scan = D3_equalize_adapthist(ct_scan, clip_limit=0.05, nbins=1000)
    elif normalize=='ZSCORE':
        ct_scan = (ct_scan - np.mean(ct_scan))/(np.std(ct_scan))
    elif normalize=='MAX':
        ct_scan = ct_scan / np.max(ct_scan)
    elif normalize=="tanh":
        ct_scan = ct_scan/np.max(ct_scan)
        #print('ct_scan',np.max(ct_scan))
        ct_scan = (ct_scan - np.mean(ct_scan))/(np.std(ct_scan)+K.epsilon())
        #print('ct_scan after z score',np.max(ct_scan))
        ct_scan = np.tanh(ct_scan) 
        #print('ct_scan after tanh',np.max(ct_scan))
        '''
        X_train_pos_ = ct_scan
        vle_min = np.min(X_train_pos_)
        vle_max = np.max(X_train_pos_)
        #X_train_pos_[np.isnan(X_train_pos_)]=0
        #X_train_pos_[np.isfinite(X_train_pos_)]=0
        for index in range(X_train_pos_.shape[0]):
            X_train_pos = X_train_pos_[index]
            mean_ = np.nanmean(X_train_pos)
            std_ = np.nanstd(X_train_pos)
            data_ = np.tanh((X_train_pos-mean_)/(std_+K.epsilon()))
            data_[np.isnan(data_),]=0
            data_[np.isfinite(data_),]=0
            X_train_pos_[index] = data_
        
        ct_scan = X_train_pos_
        '''
    return ct_scan
# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)
# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
def ApplyAugmentation(d3_img, type_of_augmentation=None, dict_parameter=None, seed=1):
    random.seed=seed
    patch_size = d3_img.shape
    d3_img_ = d3_img.reshape((patch_size[0], patch_size[1], patch_size[2]))
    if dict_parameter is None:
        dict_parameter={'rotation_xy':[-30,30],
                        'rotation_zx' :[-10,10],
                        'rotation_zy' :[-10,10],
                        'zooming':[1.0,1.1],
                        'down_scale':[0.85,0.99]
                        }
    if type_of_augmentation is None:
        seq=['None',
             'rotation_xy',
            'rotation_zx',
             'rotation_zy',
            #'zooming',
            # 'h_flip',
            #'elastic'
            #'v_flip',
            #'z_flip',
            #'down_scale'
            #'rotate_90_k1',
            # 'rotate_90_k1',
            # 'rotate_90_k2',
            # 'rotate_90_k3'
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
    new_3d_img[new_3d_img<-1.] = -1. 
    '''
    bool_val = random.choice(['T', 'F'])
    if bool_val=='T':
        new_3d_img = elastic_3d_transform(new_3d_img, seed=seed)
    '''
    return new_3d_img.reshape(patch_size)

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
        imgs_x = exposure.equalize_adapthist(img[:,:,0], clip_limit=clip_limit, nbins=nbins)
        d3_img_tmp[index,:,:,0] = imgs_x
    return d3_img_tmp
class DataGenerator(Sequence):
        def __init__(self, hdf5_path, list_IDs=None, batch_size=16, dim=(16,144,144), n_channels=1,
                 n_classes=2, shuffle=True, run_augmentations=True, type_of_augmentation=None, mode="training", convert_to_categorical=False, binarize=True, threshold_to_binary=1, Normalization=None, Two_output=False):
                self.dim = dim
                self.batch_size = batch_size
                self.mode = mode
                self.convert_to_categorical = convert_to_categorical
                self.binarize = binarize
                self.threshold_to_binary =threshold_to_binary
                self.Normalization =Normalization
                self.img_hdf5 = keras.utils.HDF5Matrix(hdf5_path, 'img')
                self.label_hdf5 = keras.utils.HDF5Matrix(hdf5_path, 'label')
                self.Two_output =Two_output
                if list_IDs==None:
                        self.list_IDs = list(range(len(self.img_hdf5)))
                else:
                        self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.n_classes = n_classes
                self.shuffle = shuffle
                self.on_epoch_end()
                self.n = len(self.list_IDs)
                self.run_augmentations = run_augmentations
                self.type_of_augmentation = type_of_augmentation
        def __len__(self):
                return int(np.floor(len(self.list_IDs) / self.batch_size))
                
        def __getitem__(self, index):
                # Generate indexes of the batch
                indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

                # Find list of IDs
                # list_IDs_temp = [self.list_IDs[k] for k in indexes]
                list_IDs_temp = [k for k in indexes]

                # Generate data
                if self.mode=="prediction":
                        X = self.__data_generation(list_IDs_temp)
                        return X
                elif self.mode=="training":
                        X, y = self.__data_generation(list_IDs_temp)
                        if self.Two_output:
                            return X, [X,y]
                        return X, y
                else:
                        raise Exception('Please select one of the modes: prediction, training')

        def show_batch(self, index):
                self.X, self.y = self.__getitem__(index)
                for i in range(self.batch_size):
                        print(self.y[i])
                        plt.imshow(self.X[i])
                        plt.show()
    
        def return_label(self, index):
                _, self.y = self.__getitem__(index)
                return self.y

        def on_epoch_end(self):
                self.indexes = np.array(self.list_IDs)
                if self.shuffle == True:
                        np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
                # Initialization
                X = np.zeros((self.batch_size, *self.dim, self.n_channels))
                y = np.zeros((self.batch_size), dtype=int)

                # Generate data
                for i, ID in enumerate(list_IDs_temp):
                        # Store sample
                        if self.Normalization is not None:
                                X[i,] = self.img_hdf5[ID].copy()
                                X[i,] = Normalize(X[i,],self.Normalization) 

                        if self.run_augmentations:
                            X[i,] = ApplyAugmentation(X[i,], self.type_of_augmentation, None) 

                        # Store class
                        if self.binarize:
                                y[i] = 0
                                if self.label_hdf5[ID]>=self.threshold_to_binary:
                                        y[i] = 1
                        else:
                                y[i] = self.label_hdf5[ID]
                if self.convert_to_categorical:
                        y=to_categorical(y, num_classes=self.n_classes)
                # Generate data
                if self.mode=="prediction":
                        return X
                elif self.mode=="training":
                        return X, y