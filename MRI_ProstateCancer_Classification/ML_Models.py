#%%
import random
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import layers
from keras.layers import Input, Conv3D, Lambda, merge, Dense, Flatten, MaxPooling3D
from keras import callbacks
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from keras.utils import to_categorical
import numpy as np
import gc
import keras
import os
import matplotlib.pyplot as plt
import keras.optimizers
from keras import regularizers
from keras import backend as K
from keras.regularizers import l2
import model_libraries
from capsule_layer import Capsule
import utils
import tensorflow as tf
gc.collect()
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]=configuration.CUDA_VISIBLE_DEVICES
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

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

#%% 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors
def D3GenerateModel(n_filter=16, number_of_class=2, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'acc'],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform', two_output=False):
    filter_size =n_filter
    input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
    x = layers.Conv3D(filters=filter_size//2, kernel_size=(1,3,3), strides=(1,1, 1), padding='same', activation='selu')(input_x)
    
    x_1_3_1 = layers.Conv3D(filters=filter_size//2, kernel_size=(1,3,1), strides=(1, 1,1), padding='same', activation='selu')(x)
    x_1_1_3 = layers.Conv3D(filters=filter_size//2, kernel_size=(1,1,3), strides=(1, 1,1), padding='same', activation='selu')(x)
    x_3_1_1 = layers.Conv3D(filters=filter_size//2, kernel_size=(3,1,1), strides=(1,1,1), padding='same', activation='selu')(x)
    #32
    x = layers.concatenate([x, x_1_3_1, x_1_1_3,x_3_1_1])
    #16
    x = layers.Conv3D(filters=filter_size, kernel_size=(1,2,2), strides=(1, 2,2), 
                                padding='same', activation='softmax')(x)
    x_0 = layers.Conv3D(filters=number_of_class, kernel_size=(1,3,3), strides=(1, 3,3), 
                                padding='same', activation='softmax')(x)
    x_1 = layers.Conv3D(filters=number_of_class, kernel_size=(3,1,3), strides=(1, 3,3), 
                                padding='same', activation='softmax')(x)
    x_2 = layers.Conv3D(filters=number_of_class, kernel_size=(3,3,1), strides=(1, 3,3), 
                                padding='same', activation='softmax')(x)
    x_3 = layers.Conv3D(filters=number_of_class, kernel_size=(3,3,3), strides=(1, 3,3), 
                                padding='same', activation='softmax')(x)
    x_output = layers.average([x_0,x_1,x_2,x_3])
    x = layers.Flatten()(x_output)
    #Encoder
    x = layers.Dense(2048, activation='selu')(x)
    x = layers.Dropout(dropout)(x)
    x= layers.Dense(1028, activation='selu')(x)
    x = layers.Dropout(dropout)(x)
    y= layers.Dense(number_of_class, activation=activation_last)(x)
    if two_output:
        model = Model(inputs=input_x, outputs=[y,x_output])
    else:
        model = Model(inputs=input_x, outputs=y)
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(lr=2e-6),loss='categorical_crossentropy', metrics=metrics)
    return model

def D3GenerateModel_old(n_filter=64, number_of_class=1, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'acc', auc],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    filter_size =16
    model = Sequential()
    #1 layer
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
#%%
# split into input (X) and output (Y) variables
# create model

cov_model = model_libraries.ConvModel_V2()
cov_model.activation_last='sigmoid'
cov_model.input_shape = (48,48)
cov_model.batch_size=10
cov_model.epoch=150
cov_model.lr=0.001
cov_model.lr_factor=2
cov_model.metrics=['acc', 'mse']
cov_model.patience = 5
cov_model.loss = 'mse'
cov_model.number_of_class=1
cov_model.number_of_channel=6
#%%
#%%
# evaluate using 10-fold cross validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    yPred = yPred>0.5
    Y_true = Y>0.5
    return (accuracy_score(Y_true, yPred), 
            precision_score(Y_true, yPred, pos_label=1, average='macro'), 
            recall_score(Y_true, yPred, pos_label=1, average='macro'))


def Normalize(data):
    X_train_pos_ = data
    for index in range(X_train_pos_.shape[0]):
        X_train_pos = X_train_pos_[index]
        X_train_pos = np.log2(X_train_pos+1) 
        X_train_pos = (X_train_pos - np.mean(X_train_pos))/ np.std(X_train_pos)
        min_p = -(np.max(X_train_pos))
        X_train_pos[X_train_pos<min_p] = 0.
        X_train_pos_[index] = X_train_pos
    return X_train_pos_
def my_scorer(estimator, x, y):
    a, p, r = getScores(estimator, x, y)
    print (a, p, r)
    return a+p+r
from skimage import exposure
def D3ImageGenerator(X,Y, batch_size=16):
    steps=X[0].shape[0]//batch_size
    X_ = X[0]
    X_1 = X[1]
    while 1:
        for i in range(0,steps):
            start_index = i*batch_size
            end_index = (i+1)*batch_size
            if start_index>(X_.shape[0]-16):
                start_index = X_.shape[0]-16
                end_index = X_.shape[0]
            
            x_batch_tmp_0 = X_[start_index:end_index]
            x_batch_tmp_1 = X_1[start_index:end_index]
            y_batch_tmp = Y[start_index:end_index]
            
            yield [x_batch_tmp_0,x_batch_tmp_1], y_batch_tmp

from keras.preprocessing.image import ImageDataGenerator
train_data_generator = ImageDataGenerator()
valid_data_generator = ImageDataGenerator()
#%%
X_train = np.load('img_train_data_3d_t2_tse_tra.npy', mmap_mode='r+')
X_valid = np.load('img_valid_data_3d_t2_tse_tra.npy', mmap_mode='r+')

Y_train = np.load('outcome_train_data_3d_t2_tse_tra.npy', mmap_mode='r+')
Y_valid = np.load('outcome_valid_data_3d_t2_tse_tra.npy', mmap_mode='r+')
#Normalize
X_train = Normalize(X_train)
X_valid = Normalize(X_valid)
#Functions
def Generate_positive_negative_lists(Y, Positive_Value=True, Negative_Value=False):
    
    Y_neg = list(np.where(Y==Negative_Value)[0].flatten())
    Y_pos = list(np.where(Y==Positive_Value)[0].flatten())
    random.shuffle(Y_neg)
    random.shuffle(Y_pos)
    return Y_pos, Y_neg
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
    return (X_all_0,X_all_1), Y_all

test = GeneralModel()
test.summary()
#Generate the positive and negative list...
Y_train_positive, Y_train_negative = Generate_positive_negative_lists(Y_train)
Y_valid_positive, Y_valid_negative = Generate_positive_negative_lists(Y_valid)
#Generate data based on the lists
x_train_data,y_train_data  = Generate_X_Y(X_train,Y_train_negative, Y_train_positive)

x_valid_data,y_valid_data  = Generate_X_Y(X_valid,Y_valid_negative, Y_valid_positive)

print('######'*30)
print('y_valid_data',y_valid_data.shape)
print('y_train_data',y_train_data.shape)

#Call model

model = D3_2Input_GenerateModel()#KerasClassifier(build_fn=GenerateModel, epochs=20, batch_size=32, verbose=1)
model.summary()
model.count_params()

log = callbacks.CSVLogger('./g_log.csv')
tb = callbacks.TensorBoard(log_dir='./tensorboard-logs',
                                batch_size=16, histogram_freq=False)
    
lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 1e-6 * (0.9 ** epoch))
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode="min", factor=0.5, patience=5, min_lr=1e-9,verbose=1)
        
checkpoint = callbacks.ModelCheckpoint('./weights-{epoch:02d}.h5', monitor='val_loss', #monitor='val_ucent_jaccard_distance',
                                            save_best_only=True, save_weights_only=True, mode='min',verbose=1)
model.fit_generator(D3ImageGenerator(x_train_data,y_train_data, 16),
                    steps_per_epoch=y_train_data.shape[0] // 16, 
                    epochs=20,
                    validation_data=D3ImageGenerator(x_valid_data,y_valid_data, 16),
                    validation_steps=y_valid_data.shape[0]//16,
                    callbacks=[log, tb, checkpoint,lr_decay, reduce_lr])

xs_x = model.predict(X)
get_cuttoff_ = np.median(xs_x)
xs = model.predict(X_test)
print(xs, Y_test)
from sklearn.metrics import roc_auc_score
vg= roc_auc_score(Y_test, xs>=get_cuttoff_)
print(vg)
'''
#%%

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(model, X, X[...,0], cv=kfold, scoring='roc_auc')
print(results.mean())

#%%
print(results)

#%%3x2x4x4x4x3=
optimizers = ['sgd']
init = ['glorot_uniform']
loss = ['mse'] #'hinge', 'poisson'
epochs = [100]
activation_last=['linear'] #, 'softmax','relu',
batches = [32] #32
Dropout = [0.2]

param_grid = dict(optimizer=optimizers, 
                  epochs=epochs, 
                  batch_size=batches, 
                  activation_last=activation_last,
                  loss=loss,
                  dropout=Dropout,
                  init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring='roc_auc')
grid_result = grid.fit(X, X[...,0])
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

#%%
'''