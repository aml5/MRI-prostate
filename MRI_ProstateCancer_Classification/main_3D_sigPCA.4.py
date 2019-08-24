
#%%
import random
from augment import augment
from scipy import ndimage
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import layers
from keras import optimizers
from keras.layers import Input, Conv3D, Lambda, merge, Dense, Flatten, MaxPooling3D
from keras import callbacks
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.utils import to_categorical
import numpy as np
from keras import initializers
import scipy
import gc
import keras
import os
import math
from convaware import ConvolutionAware
from keras_gradient_noise import add_gradient_noise
#from skimage import exposure
import cyclical_learning_rate
import matplotlib.pyplot as plt
import keras.optimizers
from keras import regularizers
from keras import backend as K
from keras.regularizers import l2
import model_libraries
from capsule_layer import Capsule
import utils
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
lr=5e-4
epochs = 30
decay = lr/epochs
counter_s=0
gc.collect()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]='1'# configuration.CUDA_VISIBLE_DEVICES
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
from tensorflow.python import debug as tf_debug
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
set_session(sess)

config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
#Metrics and Loss functions
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
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))#, np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))#, np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

def elastic_3d_transform(d3_img, verbose=False, seed=1):
    #random.seed(None)
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
    d3_img = d3_img.reshape((16,144,144))
    if dict_parameter is None:
        dict_parameter={'rotation_xy':[-10,10],
                        'rotation_zx' :[-10,10],
                        'rotation_zy' :[-10,10],
                        'zooming':[1.01,1.05],
                        #'down_scale':[0.85,0.99]
                        }
    if type_of_augmentation is None:
        seq=['None',
            'rotation_xy',
            'rotation_zx',
            'rotation_zy',
            'zooming',
            #'h_flip',
            #'v_flip',
            #'z_flip',
            #'rotate_90_k1',
            #'rotate_90_k2',
            #'rotate_90_k3'
            ]
        type_of_augmentation = random.choice(seq)

    if type_of_augmentation=='rotation_xy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(1,2),reshape=False)
    elif type_of_augmentation=='rotation_zx':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(0,2),reshape=False)
    elif type_of_augmentation=='rotation_zy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(0,1),reshape=False)
    elif type_of_augmentation=='zooming':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img, (1, value_factor, value_factor))
        x_a = new_img_zoom.shape[2]//2 - 72
        y_a = new_img_zoom.shape[1]//2 - 72
        new_3d_img = new_img_zoom[:,y_a:y_a+144, x_a:x_a+144]
    elif type_of_augmentation=='down_scale':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img, (1, value_factor, value_factor))
        new_img_zoom_tmp = np.zeros_like(d3_img)
        x_a = new_img_zoom.shape[2]//2
        y_a = new_img_zoom.shape[1]//2
        x_a_b = new_img_zoom_tmp.shape[2]//2 - x_a
        y_a_b = new_img_zoom_tmp.shape[1]//2 - y_a
        new_img_zoom_tmp[:,y_a_b:y_a_b+new_img_zoom.shape[1], x_a_b:x_a_b+new_img_zoom.shape[1]] = new_img_zoom
        new_3d_img = new_img_zoom_tmp.copy()
    elif type_of_augmentation == 'h_flip':
        new_3d_img = np.flip(d3_img,axis=1)
    elif type_of_augmentation == 'v_flip':
        new_3d_img = np.flip(d3_img,axis=2)
    elif type_of_augmentation == 'z_flip':
        new_3d_img = np.flip(d3_img,axis=0)
    elif type_of_augmentation=='rotate_90_k1':
        new_3d_img = np.rot90(d3_img,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k2':
        new_3d_img = np.rot90(d3_img,k=2,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k3':
        new_3d_img = np.rot90(d3_img,k=3,axes=(1,2))
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
        new_3d_img = d3_img
    
    bool_val = random.choice(['T', 'F'])
    if bool_val=='T':
        new_3d_img = elastic_3d_transform(new_3d_img, seed=seed)
    
    return new_3d_img.reshape((16,144,144,1))

def step_decay(epoch):
	initial_lrate = lr
	drop = 0.5
	epochs_drop = 2.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
    
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

def loss_with_uncertainty(y_true, y_pred):
	return K.mean((y_pred - y_true)**2. * K.exp(K.std(y_pred)))

def own_loss(y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    loss = 1 - ((y_pred * K.log(y_true+0.01)/(y_pred+1.01)) + y_pred)
    loss = K.sum(loss, -1)
    return loss
def postiive_input_x(y_true, y_pred):
    K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())
    return K.sum(K.argmax(y_true, axis=-1))
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.round(y_true_f)
    y_pred_f = K.round(y_pred_f)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true_f,-1) + K.sum(y_pred_f,-1) + smooth)

def recall_at_thresholds(y_pred, y_true,threshold=[0.5]):
    value, update_op = tf.metrics.recall_at_thresholds(y_pred, y_true, threshold)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall_at_thresholds' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def precision_at_thresholds(y_pred, y_true, threshold=[0.5]):
    value, update_op = tf.metrics.precision_at_thresholds(y_pred, y_true, threshold)
    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'precision_at_thresholds' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def max_margin_loss(y_true, y_pred, threshold=0.5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.mean(K.maximum(1. - (y_true_f - y_pred_f), 0.), axis=-1)

def rank_svm_objective( y_true, y_pred, margin=1.0): # change to 1.0?  makes more sense for normalized cosine distance [-1,1]
    ''' This only works when y_true and y_pred are stacked in a way so that
    the positive examples take up the first n/2 rows, and the corresponding negative samples
    take up the last n/2 rows.

    y_true corresponds to scores (e.g., inner products)
    y_pred corresponds is a vector of ones or zeros (denoting positive or negative sample)
    '''
    n = y_true.shape[0]//2
    signed = y_pred * y_true # make y_true part of the computational graph
    pos = signed[:n]
    neg = signed[n:]
    # negative samples are multiplied by -1, so that the sign in the rankSVM objective is flipped
    hinge_loss = K.relu( margin - pos - neg )
    loss_vec = K.concatenate([hinge_loss, hinge_loss], axis=0) 
    return loss_vec

def auc_loss(y_true, y_pred):
    return 1-roc_auc_score(y_true, y_pred)

def roc_auc_score(y_pred, y_true):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p)) 
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P

def weighted_binary_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred * weights), axis=-1)
    return loss

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

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

def NormalizeOnce(data):
    X_train_pos_ = data
    for index in range(X_train_pos_.shape[0]):
        X_train_pos_[index] = Normalize(X_train_pos_[index])
    return X_train_pos_
#%% 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#General Model
def AutoEncoder(x,init):
    x_a = layers.Conv3D(filters=8, kernel_size=(5,5,5), strides = (1,1,1), kernel_initializer=init, padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    x_a = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x_a)
    #36x4
    x_a = layers.Conv3D(filters=4, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    x_a = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x_a)
    #18x4
    x_a = layers.Conv3D(filters=1, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    x_a_Con = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x_a)
    #36x4
    x_a = layers.UpSampling3D(size=(1,2,2))(x_a_Con)
    x_a = layers.Conv3D(filters=4, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    #72x8
    x_a = layers.UpSampling3D(size=(2,2,2))(x_a)
    x_a = layers.Conv3D(filters=6, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    #144x16
    x_a = layers.UpSampling3D(size=(2,2,2))(x_a)
    x_a = layers.Conv3D(filters=8, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init,kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    x_a = cyclical_learning_rate.SineReLU()(x_a)
    x_a = layers.Conv3D(filters=1, kernel_size=(5,5,5), strides=(1,1, 1), padding='same',kernel_initializer=init,kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a)
    return layers.Activation('tanh',  name='autoencoder')(x_a), x_a_Con

from keras.engine.topology import Layer
from keras.constraints import min_max_norm

def CC_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1-r

class RotationThetaWeightLayer(Layer): # a scaled layer
    def __init__(self,**kwargs):
        super(RotationThetaWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W1 = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=1, max_value=4))
        self.W2 = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True,
                                      constraint=min_max_norm(min_value=1, max_value=4))
        super(RotationThetaWeightLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        
        return K.cos(3.14159265359/self.W1) * (-2) * K.exp(-(a**2+b**2)) + K.sin(3.14159265359/self.W2) * (-2) * b * K.exp(-(a**2+b**2))

def D3GenerateModel(n_filter=16, number_of_class=1, input_shape=(16,144,144,1),activation_last='softmax', metrics=['mse', 'binary_accuracy', dice_coef], loss=[CC_loss, 'binary_crossentropy'], dropout=0.05, init='glorot_uniform', two_output=False):
    #Extract unique structures
    input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
    loss_opt,core_x = AutoEncoder(input_x, init)
    #Normalize input
    input_x_t = layers.Lambda(lambda x: ((x+1)/2))(input_x)
    #Normalize Encoder data, decoder data 
    loss_opt_t = layers.Lambda(lambda x: ((x+1)/2))(loss_opt)
    
    #Generate mask.
    y_auto = layers.UpSampling3D(size=(4,8,8))(core_x)
    mask = Lambda(lambda x:K.greater(x, K.mean(x)*0.9))(y_auto)
    mask = Lambda(lambda x:K.cast(x, 'float32'))(mask)
    
    #Mask
    input_x_centered = layers.Lambda(lambda x: ((x[0]-x[1])*x[2]), name='input_centered_by_autoencoder')([input_x_t,loss_opt_t,mask])
    dist_img = layers.Lambda(lambda x: (K.abs(x[0]-x[1]) / (K.abs(x[0]*x[2])+1)) * (x[3]), name='input_centered_and_divided')([input_x,loss_opt,y_auto,mask])
    dist_img = layers.BatchNormalization(name='input_centered_and_divided_normalized')(dist_img)
    
    #INVERSE
    mask_inverse = Lambda(lambda x:1-K.cast(x, 'float32'))(mask)
    input_x_centered_inverse = layers.Lambda(lambda x: ((x[0]-x[1])*x[2]), name='input_centered_by_autoencoder_inverse')([input_x_t,loss_opt_t,mask_inverse])
    
    dist_img_inverse = layers.Lambda(lambda x: (K.abs(x[0]-x[1]) / (K.abs(x[0]*x[2])+1)) * (x[3]), name='dist_and_divided')([input_x,loss_opt,y_auto,mask_inverse])
    dist_img_inverse = layers.BatchNormalization(name='input_centered_and_divided_normalized_inverse')(dist_img_inverse)
    
    stroma_inverse = layers.Lambda(lambda x: ((1-(x[0]*x[1]))**(x[2])), name='stroma_inverse')([input_x_t,y_auto,mask_inverse])
    
    #B
    y_auto = layers.Lambda(lambda x: (x-K.min(x))/(K.max(x)-K.min(x)), name='auto_encoder')(y_auto)
    stroma = layers.Lambda(lambda x: ((1-(x[0]*x[1]))**(x[2])), name='stroma')([input_x_t,y_auto,mask])
    
    
    mixture = RotationThetaWeightLayer(name='RotationInvariant')([input_x_centered, input_x_centered_inverse])
    mixture = layers.Lambda(lambda x: ((x-K.min(x))/(K.max(x)-K.min(x))), name='RotationInvariant_min_max')(mixture)
    
    mixture_stroma = RotationThetaWeightLayer(name='RotationInvariant_STROMA')([stroma, stroma_inverse])
    mixture_stroma = layers.Lambda(lambda x: ((x-K.min(x))/(K.max(x)-K.min(x))), name='RotationInvariant_STROMA_min_max')(mixture_stroma)
    
    mixture_dist = RotationThetaWeightLayer(name='RotationInvariant_DIST')([dist_img, dist_img_inverse])
    mixture_dist = layers.Lambda(lambda x: ((x-K.min(x))/(K.max(x)-K.min(x))), name='RotationInvariant_DIST_min_max')(mixture_dist)
    
    x = layers.concatenate([dist_img, stroma])
    #16x144x144x1--> 8x72x72x16
    filter_size = n_filter
    x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides = (1,1,1), kernel_initializer=init, padding='same')(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.AveragePooling3D(pool_size=(2,2,2), padding='same')(x)

    #8x72x72x16 --> 4x36x36x32
    conv_list = []
    counter = 0
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init, kernel_regularizer=regularizers.l2(0.001))(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.AveragePooling3D(pool_size=(2,2,2), padding='same')(x)
    #3 Low-feature 4x36x36 --> 3 * 3 * 2 * *2
    for index ,kernel_sizes in enumerate([
                                [(1,3,3), (2,3,3)], #Changed [(1,3,3), (1,1,3)]
                                [(1,1,3), (2,3,3)], #Changed [(3,3,3), (3,1,3)]
                                [(1,3,1), (2,3,3)] #Changed [(3,3,1), (1,3,1)]
                                ]):
        for kernel_size in (kernel_sizes):
            x = layers.Conv3D(filters=(filter_size), kernel_size=kernel_size, 
                                kernel_initializer=init, strides =(1,1,1), 
                                padding='same', name='Conv3D_%s' % (counter)
                                )(x)

            counter = counter+1
        conv_list.append(x)
    
    #4x36x36
    x = layers.add(conv_list)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(1,1,1), strides=(1,1, 1), kernel_initializer=init,
                        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    reduce_filter =filter_size*2
    x = layers.Reshape(target_shape=[4,-1, reduce_filter])(x)

    #%%
    import math
    def primeFactors(n): 
        numbers = []
        # Print the number of two's that divide n 
        while n % 2 == 0:
                numbers.append(2)
        
                n = n / 2
          
        # n must be odd at this point 
        # so a skip of 2 ( i = i + 2) can be used 
        for i in range(3,int(math.sqrt(n))+1,2): 
          
        # while i divides n , print i ad divide n 
                while n % i== 0: 
                        numbers.append(i)
                        n = n / i
              
        # Condition if n is a prime 
        # number greater than 2 
        if n > 2: 
                numbers.append(n)
        return  numbers
    x = primeFactors(1296)
    x000 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,1296), kernel_initializer=init, strides=(1,1296))(x) #1296
    x000 = layers.BatchNormalization()(x000)
    x000 = layers.Lambda(lambda x: softmax(x))(x000)

    # A Python program to print all  
    # permutations of given length 
    from itertools import permutations, combinations,combinations_with_replacement
    
    # Get all permutations of length 2 
    # and length 2 
    unique_num = np.unique(x)
    perm = combinations_with_replacement(unique_num, 4) 
    # Print the obtained permutations 
    combination_list = []
    
    ds = list(perm)
    for i in range(0,len(ds)//2):
            combination_list.append([ds[i], ds[len(ds)-(i+1)]])
    combination_list += [[ds[len(ds)//2], ds[len(ds)//2]]]
    list_of_conv2D = []
    list_of_conv2D.append(x000)
    for comb in combination_list:
        comb_1 = comb[0]
        comb_2 = comb[1]
        from functools import reduce
        kernel_size_1 = (1, reduce(lambda x, y: x*y, comb_1))
        kernel_size_2 = (1, reduce(lambda x, y: x*y, comb_2))
        x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_1, kernel_initializer=init, strides=kernel_size_1)(x) #1296
        x00 = layers.BatchNormalization()(x00)
        x00 = layers.Activation('relu')(x00)
        x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_2, kernel_initializer=init, strides=kernel_size_2)(x00) #1296
        x00 = layers.BatchNormalization()(x00)
        x00 = layers.Lambda(lambda x: softmax(x))(x00)
        list_of_conv2D.append(x00)

    #%%%
    
    '''
    x00 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,81), kernel_initializer=init, strides=(1,81))(x) #1296
    x00 = layers.BatchNormalization()(x00)
    x00 = layers.Activation('relu')(x00)
    x00 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,16), kernel_initializer=init, strides=(1,16))(x00) #1296
    x00 = layers.BatchNormalization()(x00)
    x00 = layers.Lambda(lambda x: softmax(x))(x00)

    x01 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,36), kernel_initializer=init, strides=(1,36))(x) #1296
    x01 = layers.BatchNormalization()(x01)
    x01 = layers.Activation('relu')(x01)
    x01 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,36), kernel_initializer=init, strides=(1,36))(x01) #1296
    x01 = layers.BatchNormalization()(x01)
    x01 = layers.Lambda(lambda x: softmax(x))(x01)

    x03 = layers.Conv2D(filters=reduce_filter, kernel_size=(2,), kernel_initializer=init, strides=(1,48))(x) #1296
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Activation('relu')(x03)
    x03 = layers.Conv2D(filters=reduce_filter, kernel_size=(2,27), kernel_initializer=init, strides=(1,27))(x03) #1296
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Lambda(lambda x: softmax(x))(x03)

    x02 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,48), kernel_initializer=init, strides=(1,48))(x) #1296
    x02 = layers.BatchNormalization()(x02)
    x02 = layers.Activation('relu')(x02)
    x02 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,27), kernel_initializer=init, strides=(1,27))(x02) #1296
    x02 = layers.BatchNormalization()(x02)
    x02 = layers.Lambda(lambda x: softmax(x))(x02)

    x03 = layers.Conv2D(filters=reduce_filter, kernel_size=(2,48), kernel_initializer=init, strides=(1,48))(x) #1296
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Activation('relu')(x03)
    x03 = layers.Conv2D(filters=reduce_filter, kernel_size=(2,27), kernel_initializer=init, strides=(1,27))(x03) #1296
    x03 = layers.BatchNormalization()(x03)
    x03 = layers.Lambda(lambda x: softmax(x))(x03)
    '''
    x = layers.add(list_of_conv2D)#[x000,x00,x01,x02])
    x = Lambda(lambda x:x[0]/x[1])([x, len(list_of_conv2D)])
    
    x = layers.Reshape(target_shape=[reduce_filter,-1])(x)
    x = layers.Conv1D(filters=number_of_class, kernel_size=reduce_filter, strides=reduce_filter, kernel_initializer=init)(x)
    x = layers.Activation('sigmoid')(x)
    y = layers.Flatten(name='prediction')(x)
    #Classification
    model = Model(inputs=input_x, outputs=[loss_opt,y])
    import yogi #yogi.Yogi(lr=lr) # 
    optimizer = yogi.Yogi(lr=lr) #optimizers.adam(lr=lr, amsgrad=True)#yogi.Yogi(lr=lr) 
    model.compile(optimizer=optimizer,loss=loss, metrics={'prediction' :metrics, 'autoencoder': 'mse'}, loss_weights=[0.3,1.])#categorical_crossentropy loss_weights=[0.3,1.]
    return model
    
def D3GenerateModel_backup_27_02_2019(n_filter=16, number_of_class=1, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'binary_accuracy', dice_coef], loss=['mse','mse'], dropout=0.05, init='glorot_uniform', two_output=False):
    #Extract unique structures
    input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
    loss_opt,core_x = AutoEncoder(input_x, init)
    #Normalize input
    input_x_t = layers.Lambda(lambda x: ((x+1)/2))(input_x)
    #Normalize Encoder data, decoder data 
    #core_x = layers.Lambda(lambda x: (1-(x+1)/2)(core_x)
    loss_opt_t = layers.Lambda(lambda x: ((x+1)/2))(loss_opt)
    
    y_auto = layers.UpSampling3D(size=(4,8,8))(core_x)
    mask = Lambda(lambda x:K.less_equal(x, K.mean(x)*0.90))(y_auto)
    mask = Lambda(lambda x:K.cast(x, 'float32'))(mask)
    
    #dist_b = layers.Lambda(lambda x:(1+x[0])/((x[1]+1), name='distance')([input_x_t,loss_opt_t])
    #dist = layers.Lambda(lambda x: ((x[1]-x[0])**2 / ((x[0]*x[1])+1))*x[2], name='distance')([input_x,loss_opt_t,mask])
    #dist = layers.Lambda(lambda x: ((x[0]-x[1])**2 / (3-K.abs(x[1]+x[0])))*x[2], name='distance')([input_x_t,loss_opt_t,mask])
    input_x_centered = layers.Lambda(lambda x: ((x[0]-x[1])*x[2]), name='input_centered_by_autoencoder')([input_x_t,loss_opt_t,mask])
    dist_img = layers.Lambda(lambda x: (K.abs(x[0]-x[1]) / (K.abs(x[0]*x[2])+1)) * x[3], name='input_centered_and_divided')([input_x,loss_opt,y_auto,mask])
    dist_img = layers.BatchNormalization(name='input_centered_and_divided_normalized')(dist_img)
    
    y_auto = layers.Lambda(lambda x: 1-((x-K.min(x))/(K.max(x)-K.min(x))), name='auto_encoder')(y_auto)
    #y_auto = layers.BatchNormalization()(y_auto)
    #y_auto =  layers.Lambda(lambda x: (x-K.min(x))/(K.max(x)-K.min(x)), name='auto_encoder')(y_auto)
    stroma = layers.Lambda(lambda x: ((1-(x[0]*x[1]))**(x[2])), name='stroma')([input_x_t,y_auto,mask])
    #stroma = layers.Lambda(lambda x: 1-((x[0])*x[1])**(x[2]), name='stroma')([input_x_t,mask, y_auto])
    #x = layers.Lambda(lambda x: (1-x), name='inverse_distance')(dist)
    #%%
    #x = layers.Lambda(lambda x: (x[0]*x[1]*x[2]+K.epsilon)/(K.epsilon+x[]**(1-x[3]))))(input_x,y_auto,countour_c, loss_opt_t)
    
    #(imgs[case_id,i,:,:,0]*
    # ((conv_outputs_1[0,i,:,:,0])*
    # conv_outputs_0[0,i,:,:,0]))
    # 
    # (imgs[case_id,i,:,:,0]**(1-autoencoder_x[0,i,:,:,0]))
    
    
    
    x = layers.concatenate([dist_img, input_x_centered, stroma])
    #16x144x144x1--> 8x72x72x16
    filter_size = n_filter
    x = layers.Conv3D(filters=filter_size, kernel_size=(2,14,14), strides = (1,1,1), kernel_initializer=init, padding='same')(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size, kernel_size=(2,14,14), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

    #8x72x72x16 --> 4x36x36x32
    conv_list = []
    counter = 0
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,7,7), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,7,7), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    #3 Low-feature 4x36x36 --> 3 * 3 * 2 * *2
    for index ,kernel_sizes in enumerate([
                                [(1,7,7), (2,7,7)], #Changed [(1,3,3), (1,1,3)]
                                [(1,1,7), (2,7,7)], #Changed [(3,3,3), (3,1,3)]
                                [(1,7,1), (2,7,7)] #Changed [(3,3,1), (1,3,1)]
                                ]):
        for kernel_size in (kernel_sizes):
            x = layers.Conv3D(filters=(filter_size), kernel_size=kernel_size, kernel_initializer=init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
            counter = counter+1
        conv_list.append(x)
    
    #4x36x36
    x = layers.add(conv_list)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(1,1,1), strides=(1,1, 1), kernel_initializer=init,
                        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    reduce_filter =filter_size*2
    x = layers.Reshape(target_shape=[4,-1, reduce_filter])(x)
    x = layers.Conv2D(filters=reduce_filter, kernel_size=(1,1296), kernel_initializer=init, strides=(1,1296))(x) #1296
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    
    x = layers.Reshape(target_shape=[reduce_filter,-1])(x)
    x = layers.Conv1D(filters=number_of_class, kernel_size=reduce_filter, strides=reduce_filter, kernel_initializer=init)(x)
    x = layers.Activation('sigmoid')(x)
    y = layers.Flatten(name='prediction')(x)
    #Classification
    model = Model(inputs=input_x, outputs=[loss_opt,y])
    import yogi #yogi.Yogi(lr=lr) # 
    optimizer = yogi.Yogi(lr=lr) #optimizers.Adam(lr=lr, amsgrad=True)#
    model.compile(optimizer=optimizer,loss=loss, metrics={'prediction' :metrics, 'autoencoder': 'mse'}, loss_weights=[0.3,1.])#categorical_crossentropy
    return model





def D3GenerateModel_v2(n_filter=16, number_of_class=1, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'acc', dice_coef, recall_at_thresholds, precision_at_thresholds], loss=['mse','binary_crossentropy'], dropout=0.05, init='glorot_uniform', two_output=False):
    #init = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    filter_size =n_filter
    input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
    #Autoencoder
    #144/72/36/18
    #72x8
    
    #Section
    #1 level 16x144x144 -> 8x72x72
    #x_a_Con = layers.Conv3D(filters=1, kernel_size=(1,1,1))(x_a_Con)

    #x_a_Con = layers.UpSampling3D(size=(4,8,8))(x_a_Con)
    x = layers.multiply([input_x,y_auto])
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=filter_size, kernel_size=(4,14,14), strides = (1,1,1), kernel_initializer=init, padding='same')(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size, kernel_size=(4,14,14), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

    #2 level 8x72x72 --> 4x36x36
    conv_list = []
    counter = 0
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,7,7), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,7,7), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

    #3 Low-feature 4x36x36 --> 3 * 3 * 2 * *2
    for index ,kernel_sizes in enumerate([
                                [(1,6,6), (2,6,6)], #Changed [(1,3,3), (1,1,3)]
                                [(1,1,6), (2,6,6)], #Changed [(3,3,3), (3,1,3)]
                                [(1,6,1), (2,6,6)] #Changed [(3,3,1), (1,3,1)]
                                ]):
        for kernel_size in (kernel_sizes):
            x = layers.Conv3D(filters=(filter_size*4), kernel_size=kernel_size, kernel_initializer=init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
            #x = layers.BatchNormalization()(x)
            #x = cyclical_learning_rate.SineReLU()(x)
            counter = counter+1
        conv_list.append(x)
    #4x18x18
    x = layers.add(conv_list)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*4, kernel_size=(1,6,6), strides=(1,1, 1), kernel_initializer=init,
                        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
    reduce_filter =filter_size*4
    x = layers.Reshape(target_shape=[4,-1, reduce_filter])(x)
    x = layers.Conv2D(filters=reduce_filter, kernel_size=(1,324), kernel_initializer=init, strides=(1,324))(x) #1296
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    
    x = layers.Reshape(target_shape=[reduce_filter,-1])(x)
    x = layers.Conv1D(filters=2, kernel_size=reduce_filter, strides=reduce_filter, kernel_initializer=init)(x)
    x = layers.Activation(activation_last)(x)
    y = layers.Flatten(name='prediction')(x)
    #Classification
    model = Model(inputs=input_x, outputs=[y_auto,y])
    import yogi
    optimizer = yogi.Yogi(lr=lr) 
    model.compile(optimizer=optimizer,loss=loss, metrics={'prediction' :metrics, 'autoencoder': 'mse'}, loss_weights=[0.3,1.])#categorical_crossentropy
    return model

def D3GenerateModel_Backup(n_filter=16, number_of_class=1, input_shape=(16,144,144,1),activation_last='softmax', metrics=['mse', 'acc', dice_coef, recall_at_thresholds, precision_at_thresholds], loss='categorical_crossentropy', dropout=0.05, init='glorot_uniform', two_output=False):
    #init = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    filter_size =n_filter
    input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
    #1 level
    x = layers.Conv3D(filters=filter_size, kernel_size=(5,5,5), strides = (1,1,1), kernel_initializer=init, padding='same')(input_x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size, kernel_size=(5,5,5), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    #2 level
    conv_list = []
    counter = 0
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(3,3,3), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*2, kernel_size=(3,3,3), strides=(1,1, 1), 
                                        padding='same',kernel_initializer=init)(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.AveragePooling3D(pool_size=(1,2,2), padding='same')(x)
    x = layers.UpSampling3D(size=(1,2,2))(x)
    for index ,kernel_sizes in enumerate([
                                [(1,3,3), (3,3,1)], #Changed [(1,3,3), (1,1,3)]
                                [(3,3,3), (3,1,3)], #Changed [(3,3,3), (3,1,3)]
                                [(3,3,1), (3,3,3), (1,3,3)] #Changed [(3,3,1), (1,3,1)]
                                ]):
        for kernel_size in (kernel_sizes):
            x = layers.Conv3D(filters=(filter_size*4), kernel_size=kernel_size, kernel_initializer=init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
            x = layers.BatchNormalization()(x)
            x = cyclical_learning_rate.SineReLU()(x)
            counter = counter+1
        conv_list.append(x)
    x = layers.concatenate(conv_list)
    x = layers.Conv3D(filters=filter_size*8, kernel_size=(3,3,3), strides=(2,2, 2), kernel_initializer=init,
                        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Conv3D(filters=filter_size*4, kernel_size=(1,1,1), strides=(1,1, 1), kernel_initializer=init,
                        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    #x = layers.MaxPooling3D(pool_size=(2,2, 2))(x)
    x = layers.Reshape(target_shape=[4,-1, filter_size*4])(x)
    x = layers.Conv2D(filters=filter_size*4, kernel_size=(1,1296), kernel_initializer=init, strides=(1,1296))(x)
    x = layers.BatchNormalization()(x)
    x = cyclical_learning_rate.SineReLU()(x)
    x = layers.Reshape(target_shape=[filter_size*4,-1])(x)
    x = layers.Conv1D(filters=2, kernel_size=filter_size*4, strides=filter_size*4, kernel_initializer=init)(x)
    x = layers.Softmax()(x)
    y = layers.Flatten()(x)
    #Classification    
    model = Model(inputs=input_x, outputs=y)
    #optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0.000001,lr=lr)
    #keras.optimizers.SGD(lr=lr, momentum=0.90, decay=decay, nesterov=False)
    #opt_noise = add_gradient_noise(optimizers.Adam)
    #optimizer = 'Adam'#opt_noise(lr, amsgrad=True)#, nesterov=True)#opt_noise(lr, amsgrad=True)
    import yogi
    optimizer = yogi.Yogi(lr=lr) 
    #optimizer=optimizers.adam(lr, amsgrad=True)
    model.compile(optimizer=optimizer,loss=loss, metrics=metrics)#categorical_crossentropy
    return model

def D3GenerateModel_old(n_filter=64, number_of_class=1, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'acc', auc],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    filter_size =16
    model = Sequential()
    #1 layer
    model.add(layers.Conv3D(filters=filter_size, input_shape=input_shape,  kernel_size=(2,2,2), strides=(1,1, 1), 
                                padding='same', activation='relu'))
    model.add(layers.Conv3D(filters=filter_size, input_shape=input_shape,  kernel_size=(2,2,2), strides=(1,1, 1), 
                                padding='same', activation='relu'))
    model.add(layers.MaxPooling3D((1, 2,2), strides=(1,2,2), padding='valid'))
    #2 layer
    for i in range(1,5):
        model.add(layers.Conv3D(filters=filter_size, kernel_size=(2,2,2), strides=(1,1,1), 
                                    padding='same', activation='relu'))
        model.add(layers.Conv3D(filters=filter_size*i, kernel_size=(2,2,2), strides=(1,1,1), 
                                    padding='same', activation='relu'))
        model.add(layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='valid'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation='linear', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.summary()
    model.compile(optimizer=keras.optimizers.sgd(lr=1e-4, nesterov=True),loss='hinge', metrics=metrics)#keras.optimizers.SGD
    return model
def GeneralModel(filter_size=16,input_shape=(16,144,144,1), activation_last='sigmoid'):
    input_ = Input(shape=input_shape)
    x = Conv3D(filters=filter_size, kernel_size=(3,3,3), strides=(1,1, 1), padding='same', activation='relu', kernel_regularizer=l2(2e-4), kernel_initializer='he_normal')(input_)
    x = MaxPooling3D((1, 2,2), strides=(1,2,2), padding='valid')(x)
    #2 layer
    for i,filter_s in enumerate([32,64,128,256]):
        x = Conv3D(filters=filter_s, kernel_size=(3,3,3), strides=(1,1,1), 
                                    padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Conv3D(filters=filter_s, kernel_size=(3,3,3), strides=(1,1,1), 
                                    padding='same', activation='relu',kernel_initializer='he_normal')(x)
        if i==2:
            x = Conv3D(filters=filter_s, kernel_size=(3,3,3), strides=(1,1,1), 
                                    padding='same', activation='relu',kernel_initializer='he_normal')(x)
        elif i==3:
            x = Conv3D(filters=filter_s, kernel_size=(3,3,3), strides=(1,1,1), 
                                    padding='same', activation='relu',kernel_initializer='he_normal')(x)
            x = MaxPooling3D((1, 4, 4), strides=(1, 4, 4), padding='valid')(x)
        else:  
            x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding='valid')(x)
    x = layers.Lambda(squash)(x)
    '''
    _sg = []
    for kernel in [(1,2,1),[1,1,1], (1,2,2), (1,1,2)]:
        x_x = Conv3D(filters=128, kernel_size=kernel, strides=(1,1,1), 
                                     padding='same', activation='relu',kernel_initializer='he_normal')(x)
        _sg.append(x_x)
    
    x = layers.concatenate(_sg)
    '''
    x = MaxPooling3D((16, 2, 2), strides=(16, 2, 2), padding='same')(x)
    '''
    x = Conv3D(filters=512, kernel_size=(1,1,1), strides=(1,1,1), 
                                    padding='same', activation=squash,kernel_initializer='he_normal')(x)
    '''
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='valid')(x)
    x = Flatten()(x) #layers.GlobalMaxPool3D()(x)
    x = Dense(1024, activation=activation_last)(x)#5184
    #x = layers.Dropout(0.3)(x)
    #x = Dense(2048, activation=activation_last)(x)
    return Model(input_, x)
def D3_2Input_GenerateModel(n_filter=16, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse',accuracy],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)
    print('positive_input',positive_input)
    base_model = GeneralModel(n_filter, input_shape)
    positive_input_z = base_model(positive_input)
    negative_input_z = base_model(negative_input)
    distance = Lambda(euclidean_distance, eucl_dist_output_shape)([positive_input_z, negative_input_z])
    #prediction = Dense(1,activation='sigmoid')(distance)
    siamese_net = Model(input=[positive_input,negative_input],output=distance)
    optimizer = keras.optimizers.Adam(1e-3)
    siamese_net.compile(loss=contrastive_loss,optimizer=optimizer, metrics=metrics)
    return siamese_net
def GenerateModel(n_filter=32, number_of_class=2, input_shape=(48,48), number_of_channel=3,activation_last='sigmoid', metrics=['mse', 'acc'],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    ########
    #
    #
    #######
    init_X = init #keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    filter_size =n_filter #n_filter*4
    model = Sequential()
    '''AutoEncoder'''
    shape_default = (input_shape[0], input_shape[1], number_of_channel)
    shape_range= input_shape[0]* input_shape[1] * 1
    model.add(layers.Conv2D(filters=filter_size, input_shape=shape_default,  kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.Conv2D(filters=filter_size, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid'))
    model.add(layers.Conv2D(filters=filter_size*2, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.Conv2D(filters=filter_size*2, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.MaxPooling2D((2, 2), padding='valid'))
    
    model.add(layers.Conv2D(filters=filter_size*4, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.Conv2D(filters=filter_size*4, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    model.add(layers.Conv2D(filters=filter_size*8, kernel_size=3, strides=(1, 1), 
                                padding='same', activation='selu'))
    #model.add(layers.MaxPooling2D((2, 2), padding='valid'))
    
    model.add(layers.GlobalMaxPooling2D())
    #Encoder
    '''
    model.add(layers.Dense(shape_range//2))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    model.add(layers.Dense(shape_range//3))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    model.add(layers.Dense(shape_range//4))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    '''
    #model.add(layers.Dense(shape_range//5))
    #model.add(keras.layers.LeakyReLU())
    #model.add(keras.layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='selu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='selu'))
    model.add(layers.Dense(2, activation='softmax'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    '''
    #Decoder
    shape_default = (input_shape[0], input_shape[1], 1)
    shape_range= input_shape[0]* input_shape[1] * 1
    filter_decoder = shape_range//6
    
    model.add(layers.Dense(filter_decoder*2))
    model.add(keras.layers.LeakyReLU())
    model.add(layers.Dense(filter_decoder*3))
    model.add(keras.layers.LeakyReLU())
    model.add(layers.Dense(filter_decoder*4))
    model.add(keras.layers.LeakyReLU())
    
    model.add(layers.Dense(shape_range, activation='sigmoid'))
    model.add(layers.Reshape(shape_default))
    '''
    model.compile(optimizer=keras.optimizers.adam(lr=2e-6),loss='categorical_crossentropy', metrics=[auc])
    return model
    '''
    model.add(layers.Conv2D(filters=8*32, kernel_size=9, strides=2, padding='valid'))
    model.add(layers.Reshape((-1, filter_size)))
    model.add(layers.Lambda(utils.squash, name='primarycap_squash'))
    model.add(Capsule(2, 32, 3, True))
    model.add(layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2))))
    
    model.add(layers.Conv2D(filters=filter_size, 
                                kernel_size = (3,3),
                                strides=2,
                                padding='same'))
    model.add(layers.BatchNormalization(scale=True))
    model.add(layers.Activation('relu'))
    for kernel_size in ((1,3), (3,1),(1,1)):
        model.add(layers.Conv2D(filters=(filter_size//2), 
                                    kernel_size = kernel_size,
                                    strides=1,
                                    padding='same'))
        model.add(layers.BatchNormalization(scale=True))
        model.add(layers.Activation('relu'))
    #model.add(layers.Dropout(0.2))
    for i in [3,4]:
        model.add(layers.Conv2D(filters=(filter_size//i), 
                                    kernel_size = (3,3),
                                    strides=1,
                                    padding='same'))
        model.add(layers.BatchNormalization(scale=True))
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D((2, 2), padding='valid'))
        
    model.add(layers.Dropout(dropout))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(number_of_class, activation=activation_last))
    
    model.compile(optimizer='adam',loss=utils.margin_loss, metrics=metrics)
    return model
    '''
#from skimage.exposure import equalize_adapthist
def D3ImageGenerator(X,Y, batch_size=16, RunNormalize=True):
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
            
            yield x_batch_tmp_0, [x_batch_tmp,y_batch_tmp]#[x_batch_tmp_0,x_batch_tmp_1], y_batch_tmp
def Generate_positive_negative_lists(Y, Positive_Value=1, Negative_Value=0):
    Y_neg = list(np.where(Y==Negative_Value)[0].flatten())
    Y_pos = list(np.where(Y==Positive_Value)[0].flatten())
    random.shuffle(Y_neg)
    random.shuffle(Y_pos)
    return Y_pos, Y_neg
def GenerateEqualPositiveAndNegativeValue_wrong(X,Y, batch_size=16, RunNormalize=True, max_iteration_per_epoch=10000, RunApplyAugmentation=False):
    P_indexes, N_indexes = Generate_positive_negative_lists(Y)
    steps=X.shape[0]//batch_size
    counter  = 0
    while 1:
        if counter>=max_iteration_per_epoch:
            counter =0
        
        for i in range(0,steps):
            random.seed(
                counter
            )
            N_indexes_randomly_selected = random.sample(N_indexes,8)
            P_indexes_randomly_selected = random.sample(P_indexes, 8)
            list_selectred = N_indexes_randomly_selected + P_indexes_randomly_selected
            convert_selected = P_indexes_randomly_selected + N_indexes_randomly_selected
            #random.shuffle(list_selectred)
            x_batch_tmp = X[list_selectred].copy()
            x_batch_tmp_c = X[convert_selected].copy()
            y = np.zeros([batch_size,2])
            y[:8,0] = 1.
            y[8:,1] = 1.
            counter += 1
            if RunNormalize:
                for index in range(x_batch_tmp.shape[0]):
                    x_batch_tmp[index] = Normalize(x_batch_tmp[index])
                    x_batch_tmp_c[index] = Normalize(x_batch_tmp_c[index])

                    if RunApplyAugmentation:
                        x_batch_tmp[index] = ApplyAugmentation(x_batch_tmp[index], seed=counter)
                        x_batch_tmp_c[index] = ApplyAugmentation(x_batch_tmp_c[index], seed=counter)
            yield x_batch_tmp, [x_batch_tmp_c,y]
            
def GenerateEqualPositiveAndNegativeValue(X,Y, batch_size=16, RunNormalize=True, max_iteration_per_epoch=10000, RunApplyAugmentation=False):
    P_indexes, N_indexes = Generate_positive_negative_lists(Y)
    counter  = 0
    while 1:
        if counter>=max_iteration_per_epoch:
            counter =0
        random.seed(counter)
        N_indexes_randomly_selected = random.sample(N_indexes,8)
        P_indexes_randomly_selected = random.sample(P_indexes, 8)
        list_selectred = N_indexes_randomly_selected + P_indexes_randomly_selected
        convert_selected = P_indexes_randomly_selected + N_indexes_randomly_selected
        #dd_f = list(range(batch_size))
        #random.shuffle(dd_f, None)
        x_batch_tmp = X[list_selectred].copy()
        x_batch_tmp_c = X[convert_selected].copy()
        y_batch_tmp = Y[list_selectred].copy()
        y_ = np.zeros([y_batch_tmp.shape[0],2])
        for i, y_vle  in enumerate(y_batch_tmp):
            y_[i, y_vle] = 1.

        counter += 1
        if RunNormalize == True or RunApplyAugmentation==True:
            for index in range(x_batch_tmp.shape[0]):
                if RunApplyAugmentation:
                    x_batch_tmp[index] = ApplyAugmentation(x_batch_tmp[index], seed=counter)
                    x_batch_tmp_c[index] = ApplyAugmentation(x_batch_tmp_c[index], seed=counter)
                if RunNormalize:
                    x_batch_tmp[index] = Normalize(x_batch_tmp[index])
                    x_batch_tmp_c[index] = Normalize(x_batch_tmp_c[index])
        dd_f = list(range(batch_size))
        random.shuffle(dd_f, None)
        x_batch_tmp_c = x_batch_tmp_c[dd_f]
        x_batch_tmp = x_batch_tmp[dd_f]
        y_=y_batch_tmp[dd_f]
        yield x_batch_tmp, [x_batch_tmp_c,y_]
            
def Generate_X_Y(X_data, negative_lst, positive_lst):
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
#%%
#MAIN FUNCTION
def Run(categorical=False, Normalize=False, shuffle=True):
    X_train = np.load('img_train_data_3d_t2_tse_tra.npy', mmap_mode='r')
    X_valid = np.load('img_valid_data_3d_t2_tse_tra.npy', mmap_mode='r')

    Y_train = np.load('outcome_train_data_3d_t2_tse_tra.npy')#, mmap_mode='r')
    Y_valid = np.load('outcome_valid_data_3d_t2_tse_tra.npy')#, mmap_mode='r')
    indexes = np.arange(X_train.shape[0])
    if shuffle == True:
        for i in range(10):
            np.random.shuffle(indexes)
        X_train = X_train[indexes]
    
    if Normalize:
        pass
    #    X_train = NormalizeOnce(X_train)
    #    X_valid = NormalizeOnce(X_valid)

    #X_train = np.nan_to_num(X_train)
    #X_valid = np.nan_to_num(X_valid)
    #print('mean X', np.mean(X_train))
    #print('mean Valid', np.mean(X_valid))

    #X_train = Normalize(X_train)
    #X_valid = Normalize(X_valid)
    from keras.utils import to_categorical
    if categorical:
        Y_train = to_categorical(Y_train.astype(int), num_classes=2)
        Y_valid =to_categorical(Y_valid.astype(int), num_classes=2)

    print('######'*30)
    print('x train_data',X_train.shape)
    print('x valid_data',X_valid.shape)

    print('y train_data',Y_train.shape)
    print('y valid_data',Y_valid.shape)
    
    #Call model
    par_batch_size=16
    model = D3GenerateModel(number_of_class=1, input_shape=(16,144,144,1))#KerasClassifier(build_fn=GenerateModel, epochs=20, batch_size=32, verbose=1)
    #del model
    #from keras.models import load_model
    #model = load_model('./result_recent/weights-47.h5', custom_objects={'dice_coef':dice_coef, 'recall_at_thresholds': recall_at_thresholds, 
    #'precision_at_thresholds' : precision_at_thresholds, 'SineReLU': cyclical_learning_rate.SineReLU

    #})
    model.summary()
    model.count_params()
    iterations = int(round(X_train.shape[0]*1)) #4
    valid_iteration = (X_valid.shape[0]*1)#*2)
    log = callbacks.CSVLogger('./result_recent/log.csv')
    tb = callbacks.TensorBoard(log_dir='./result_recent/tensorboard-logs',
                                    batch_size=par_batch_size, histogram_freq=False)
       
    lr_decay = callbacks.LearningRateScheduler(schedule=step_decay)#lambda epoch: lr * (0.9999 ** epoch))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min", factor=0.5, patience=3, min_lr=1e-10,verbose=1)
    

    clr = cyclical_learning_rate.CyclicLR(base_lr=lr, max_lr=lr*6,
                                step_size=2*iterations, Peak_Location='LowFirst')#, mode='exp_range')
    
    checkpoint = callbacks.ModelCheckpoint('./result_recent/weights-{epoch:02d}.h5', monitor='val_dice_coef', #monitor='val_ucent_jaccard_distance',
                                                save_best_only=False, save_weights_only=False, mode='max',verbose=1)
    model.fit_generator(GenerateEqualPositiveAndNegativeValue(X_train,Y_train, batch_size=16, max_iteration_per_epoch=iterations, RunApplyAugmentation=False),
                        steps_per_epoch=(iterations //par_batch_size), #X_train.shape[0]
                        epochs=500,
                        class_weight={0:1, 1:1.
                        },
                        use_multiprocessing=False,
                        workers=1,
                        shuffle=False,
                        validation_data=GenerateEqualPositiveAndNegativeValue(X_valid,Y_valid, batch_size=16, max_iteration_per_epoch=valid_iteration, RunApplyAugmentation=False),
                        validation_steps=X_valid.shape[0]//par_batch_size,
                        callbacks=[log, tb, checkpoint,clr])

Run()