import random
from keras.models import Sequential, Model
from keras.layers import Dense
from keras import layers
from keras import optimizers
from keras.layers import Input, Conv3D, Lambda, merge, Dense, Flatten, MaxPooling3D
from keras import callbacks
import numpy as np
from keras import initializers
import keras
from convaware import ConvolutionAware
import cyclical_learning_rate
import keras.optimizers
from keras import regularizers
from keras import backend as K
from keras.regularizers import l2
import utils
from utils import RotationThetaWeightLayer
import tensorflow as tf
from keras.constraints import min_max_norm
seed = 7
np.random.seed(seed)
#General Model

class Models():
    #Batch Normalization
    def Conv2DBNSLU(self, x, filters, kernel_size=1, strides=1, padding='same', activation=None, name=None, scale=True, BN=True):
        x = layers.Conv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=False)(x)
        if BN:
            x = layers.BatchNormalization(scale=scale)(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        return x
    def SepConv2DBNSLU(self, x, filters, kernel_size=1, strides=1, padding='same', activation=None, name=None, scale=True, BN=True):
        x = layers.SeparableConv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=False)(x)
        if BN:
            x = layers.BatchNormalization(scale=scale)(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        return x

    def __init__(self, input_shape=(512,512), initial_filter=4, batch_size=16, n_class=2, number_input_channel=3, compression_rate=0.5,final_activation="softmax", init="glorot_uniform", dropout=0.05, two_output=False):
        self.input_shape= input_shape
        self.n_class = n_class
        self.initial_filter=initial_filter
        self.batch_size = batch_size
        self.number_input_channel=number_input_channel
        self.final_activation=final_activation
        self.compression_rate = compression_rate
        self.init=init 
        self.dropout=dropout
        self.two_output=two_output
    

    def AutoEncoder(self, x,init, last_activation="tanh"):
        x_a = layers.Conv3D(filters=6, kernel_size=(5,16,16), strides = (1,1,1), kernel_initializer=init, padding='same', kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x) #
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        x_a = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x_a)
        #36x4
        x_a = layers.Conv3D(filters=3, kernel_size=(5,16,16), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        x_a = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x_a)
        #18x4
        x_a = layers.Conv3D(filters=1, kernel_size=(5,16,16), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        x_a_Con = layers.MaxPooling3D(pool_size=(1,2,2), padding='same',name='x_a_Con')(x_a)
        #36x4
        x_a = layers.UpSampling3D(size=(1,2,2))(x_a_Con)
        x_a = layers.Conv3D(filters=2, kernel_size=(5,16,16), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        #72x8
        x_a = layers.UpSampling3D(size=(2,2,2))(x_a)
        x_a = layers.Conv3D(filters=4, kernel_size=(5,16,16), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        #144x16
        x_a = layers.UpSampling3D(size=(2,2,2))(x_a)
        x_a = layers.Conv3D(filters=6, kernel_size=(5,16,16), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #,kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = cyclical_learning_rate.SineReLU()(x_a)
        x_a = layers.Conv3D(filters=1, kernel_size=(1,1,1), strides=(1,1, 1), padding='same',kernel_initializer=init, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x_a) #kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        return layers.Activation(last_activation,  name='autoencoder')(x_a), x_a_Con

    def D3GenerateModel_V2(self):
        #Extract unique structures
        n_filter=self.initial_filter
        init=self.init
        two_output=self.two_output
        dropout=self.dropout
        number_of_class=self.n_class
        input_shape = (self.input_shape[0],self.input_shape[1],self.input_shape[2], self.number_input_channel)
        activation_last=self.final_activation
        
        input_x = layers.Input(shape= input_shape,name='Input_layer', dtype = 'float32')
        x = input_x#layers.concatenate([input_x_centered,dist_img,stroma])#, mixture])
        #16x144x144x1--> 8x72x72x16
        filter_size = n_filter
        x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides = (1,1,1), kernel_initializer=init, padding='same')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = cyclical_learning_rate.SineReLU()(x)#layers.Activation('relu')(x)
        '''
        x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init)(x)
        x = cyclical_learning_rate.SineReLU()(x) #layers.Activation('relu')(x)#
        '''
        x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)

        #8x72x72x16 --> 4x36x36x32
        conv_list = []
        counter = 0
        x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)
        '''
        x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init)(x)
        x = cyclical_learning_rate.SineReLU()(x)
        '''
        x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
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
                x = layers.BatchNormalization()(x)
                counter = counter+1
                #x = cyclical_learning_rate.SineReLU()(x)
            conv_list.append(x)
        
        #4x36x36
        x = layers.add(conv_list)
        #x = layers.Lambda(lambda x: K.max(x, axis=-1))(x)
        #x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)

        x = layers.Conv3D(filters=filter_size*4, kernel_size=(1,1,1), strides=(1,1, 1), kernel_initializer=init,
                            padding='same')(x)
        
        x = layers.BatchNormalization(scale=False)(x)
        x = cyclical_learning_rate.SineReLU()(x)
        reduce_filter =filter_size*4
        x = layers.Reshape(target_shape=[16,-1, reduce_filter])(x)

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
        
        shape_of_ROI = x._keras_shape[2]
        primeFactors = primeFactors(shape_of_ROI)
        ##
        import Layers

        x = Layers.AttentionLayer()(x)
        x = layers.Add()([x,x])
        x000 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,shape_of_ROI), kernel_initializer=init, strides=(1,shape_of_ROI))(x) #1296
        x000 = layers.BatchNormalization()(x000)
        x000 = layers.Lambda(lambda x: utils.softmax(x))(x000)
        #x000 = Lambda(lambda x:(x-K.min(x))/(K.max(x)-K.min(x)), name='Softmax_NORMAL')(x000)
        
        # A Python program to print all  
        # permutations of given length 
        from itertools import permutations, combinations,combinations_with_replacement
        
        # Get all permutations of length 2 
        # and length 2 
        unique_num = np.unique(primeFactors)
        perm = combinations_with_replacement(unique_num,4) 
        # Print the obtained permutations 
        combination_list = []
        from functools import reduce
        ds = list(perm)
        for i in range(0,len(ds)//2):
            s = [[ds[i], ds[len(ds)-(i+1)]]]
            combination_list +=s
        combination_list += [[ds[len(ds)//2], ds[len(ds)//2]]]
        
        list_of_conv2D = []
        list_of_conv2D.append(x000)
        print('combination_list',len(combination_list))
        for comb in combination_list:
            print(comb)
            comb_1 = comb[0]
            comb_2 = comb[1]
            print(comb_2,comb_1)
            kernel_size_1 = (1, reduce(lambda x, y: x*y, comb_1))
            kernel_size_2 = (1, reduce(lambda x, y: x*y, comb_2))
            print(kernel_size_1,kernel_size_2)
            x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_1, kernel_initializer=init, strides=kernel_size_1, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x) #1296 #, kernel_constraint=min_max_norm(min_value=-1, max_value=1)
            x00 = layers.BatchNormalization(scale=False)(x00)
            x00 = layers.Lambda(lambda x: utils.squash(x))(x00)
            x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_2, kernel_initializer=init, strides=kernel_size_2, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x00) #1296 #, kernel_constraint=min_max_norm(min_value=-1, max_value=1)
            x00 = layers.BatchNormalization(scale=False)(x00)
            x00 = layers.Lambda(lambda x: utils.softmax(x))(x00)
            x00 = Lambda(lambda x:(x-K.min(x))/(K.max(x)-K.min(x)))(x00)
        
            list_of_conv2D.append(x00)

        #%%%
        x = layers.add(list_of_conv2D)#[x000,x00,x01,x02])
        x = layers.Lambda(lambda x: x/4) (x)
        x = layers.Flatten()(x)
        #x = layers.Reshape(target_shape=[reduce_filter,-1])(x)
        #x = layers.Conv1D(filters=number_of_class, kernel_size=reduce_filter, strides=reduce_filter, kernel_initializer=init)(x)
        x = layers.Dense(256, activation="relu")(x)
        y = layers.Dense(1, activation=activation_last,name='prediction')(x)
        #x = layers.Activation(activation_last)(x)
        #y = layers.Flatten(name='prediction')(x)
        #Classification
        if self.two_output:
            model = Model(inputs=input_x, outputs=[loss_opt,y])
            return model
        else:
            model = Model(inputs=input_x, outputs=y)
            return model


    def D3GenerateModel(self):
        #Extract unique structures
        n_filter=self.initial_filter
        init=self.init
        two_output=self.two_output
        dropout=self.dropout
        number_of_class=self.n_class
        input_shape = (self.input_shape[0],self.input_shape[1],self.input_shape[2], self.number_input_channel)
        activation_last=self.final_activation
        
        input_x = layers.Input(shape= input_shape,name='Input_layer', dtype = 'float32')
        loss_opt,core_x = self.AutoEncoder(input_x, init, "tanh")
        #Normalize input
        input_x_t = layers.Lambda(lambda x: ((x+1)/2))(input_x)
        #Normalize Encoder data, decoder data 
        if self.two_output:
            loss_opt_t = layers.Lambda(lambda x: ((x+1)/2))(loss_opt)
        else:
            loss_opt_t = input_x_t
        #Generate mask.
        y_auto = layers.UpSampling3D(size=(4,8,8))(core_x)
        #y_auto = Lambda(lambda x:K.l2_normalize(x), name='Y_AUTO')(y_auto)
        y_auto = Lambda(lambda x:(x-K.min(x))/(K.max(x)-K.min(x)), name='Y_AUTO_NORMAL')(y_auto)
        mask = Lambda(lambda x:K.greater_equal(x, K.mean(x)*0.9), name='MASK')(y_auto)
        mask = Lambda(lambda x:K.cast(x, 'float32'))(mask)
        mask_inverse = Lambda(lambda x:K.less(x, K.mean(x)), name='MASK_INVERSE')(y_auto)
        mask_inverse = Lambda(lambda x:K.cast(x, 'float32'))(mask)
        #Mask
        input_x_centered = layers.Lambda(lambda x: ((x[0]-x[1])*x[2]), name='input_centered_by_autoencoder')([input_x_t,loss_opt_t,mask])
        #INVERSE
        
        input_x_centered_inverse = layers.Lambda(lambda x: (x[0]-x[1])-K.mean((x[0]-x[1]))/K.std((x[0]-x[1])), name='input_centered_by_autoencoder_inverse')([input_x_t,loss_opt_t])
        dist_img = layers.Lambda(lambda x: (K.abs(x[0]-x[1]) / (K.abs(x[0]*x[2])+1)) * (x[3]), name='input_centered_and_divided')([input_x,loss_opt,y_auto,mask_inverse])
        dist_img = layers.BatchNormalization(name='input_centered_and_divided_normalized')(dist_img)
    
        #B
        y_auto_v = layers.Lambda(lambda x: 1-x, name='auto_encoder_inversed_x')(y_auto)
        stroma = layers.Lambda(lambda x: (1-(x[0]*x[1]))**x[2], name='stroma')([input_x_t,y_auto_v,mask])

        #mixture = RotationThetaWeightLayer(name='RotationInvariant')([input_x_centered, input_x_centered_inverse])
        #mixture = layers.BatchNormalization()(mixture)
        #mixture = layers.Lambda(lambda x: ((x-K.min(x))/(K.max(x)-K.min(x))), name='RotationInvariant_min_max')(mixture)
        
        x = layers.concatenate([input_x_centered,dist_img,stroma])#, mixture])
        #16x144x144x1--> 8x72x72x16
        filter_size = n_filter
        x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides = (1,1,1), kernel_initializer=init, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)#layers.Activation('relu')(x)
        '''
        x = layers.Conv3D(filters=filter_size, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init)(x)
        x = cyclical_learning_rate.SineReLU()(x) #layers.Activation('relu')(x)#
        '''
        x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)

        #8x72x72x16 --> 4x36x36x32
        conv_list = []
        counter = 0
        x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)
        '''
        x = layers.Conv3D(filters=filter_size*2, kernel_size=(2,5,5), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=init)(x)
        x = cyclical_learning_rate.SineReLU()(x)
        '''
        x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
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
                x = layers.BatchNormalization()(x)
                counter = counter+1
                #x = cyclical_learning_rate.SineReLU()(x)
            conv_list.append(x)
        
        #4x36x36
        x = layers.add(conv_list)
        #x = layers.Lambda(lambda x: K.max(x, axis=-1))(x)
        #x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)

        x = layers.Conv3D(filters=filter_size*4, kernel_size=(1,1,1), strides=(1,1, 1), kernel_initializer=init,
                            padding='same')(x)
        
        x = layers.BatchNormalization()(x)
        x = cyclical_learning_rate.SineReLU()(x)
        reduce_filter =filter_size*4
        x = layers.Reshape(target_shape=[16,-1, reduce_filter])(x)

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
        
        shape_of_ROI = x._keras_shape[2]
        primeFactors = primeFactors(shape_of_ROI)
        ##
        import Layers

        x = Layers.AttentionLayer()(x)
        x = layers.Add()([x,x])
        x000 = layers.Conv2D(filters=reduce_filter, kernel_size=(1,shape_of_ROI), kernel_initializer=init, strides=(1,shape_of_ROI))(x) #1296
        x000 = layers.BatchNormalization()(x000)
        x000 = layers.Lambda(lambda x: utils.softmax(x))(x000)
        #x000 = Lambda(lambda x:(x-K.min(x))/(K.max(x)-K.min(x)), name='Softmax_NORMAL')(x000)
        
        # A Python program to print all  
        # permutations of given length 
        from itertools import permutations, combinations,combinations_with_replacement
        
        # Get all permutations of length 2 
        # and length 2 
        unique_num = np.unique(primeFactors)
        perm = combinations_with_replacement(unique_num, 4) 
        # Print the obtained permutations 
        combination_list = []
        from functools import reduce
        ds = list(perm)
        for i in range(0,len(ds)//2):
            s = [[ds[i], ds[len(ds)-(i+1)]]]
            combination_list +=s
        combination_list += [[ds[len(ds)//2], ds[len(ds)//2]]]
        
        list_of_conv2D = []
        list_of_conv2D.append(x000)
        print('combination_list',len(combination_list))
        for comb in combination_list:
            print(comb)
            comb_1 = comb[0]
            comb_2 = comb[1]
            print(comb_2,comb_1)
            kernel_size_1 = (1, reduce(lambda x, y: x*y, comb_1))
            kernel_size_2 = (1, reduce(lambda x, y: x*y, comb_2))
            print(kernel_size_1,kernel_size_2)
            x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_1, kernel_initializer=init, strides=kernel_size_1, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x) #1296 #, kernel_constraint=min_max_norm(min_value=-1, max_value=1)
            x00 = layers.BatchNormalization()(x00)
            x00 = layers.Lambda(lambda x: utils.squash(x))(x00)
            x00 = layers.Conv2D(filters=reduce_filter, kernel_size=kernel_size_2, kernel_initializer=init, strides=kernel_size_2, kernel_constraint=min_max_norm(min_value=-1, max_value=1))(x00) #1296 #, kernel_constraint=min_max_norm(min_value=-1, max_value=1)
            x00 = layers.BatchNormalization()(x00)
            x00 = layers.Lambda(lambda x: utils.softmax(x))(x00)
            #x00 = Lambda(lambda x:(x-K.min(x))/(K.max(x)-K.min(x)))(x00)
        
            list_of_conv2D.append(x00)

        #%%%
        x = layers.add(list_of_conv2D)#[x000,x00,x01,x02])
        x = layers.Lambda(lambda x: x/4) (x)
        x = layers.Reshape(target_shape=[reduce_filter,-1])(x)
        x = layers.Conv1D(filters=number_of_class, kernel_size=reduce_filter, strides=reduce_filter, kernel_initializer=init)(x)
        x = layers.Activation(activation_last)(x)
        y = layers.Flatten(name='prediction')(x)
        #Classification
        if self.two_output:
            model = Model(inputs=input_x, outputs=[loss_opt,y])
            return model
        else:
            model = Model(inputs=input_x, outputs=y)
            return model

    def DataInspector3D(self):
        shape_default  = (self.input_shape[0], self.input_shape[1],  self.input_shape[2],self.number_input_channel)
        x = layers.Input(shape=shape_default)
        #x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x)

        filter_size = self.initial_filter
        x_y_o = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides = (1,1,1), padding='same')(x)
        x_y_o = layers.BatchNormalization(scale=False)(x_y_o)
        y = self.Spider_3D_Net(x_y_o, initial_filter = 2,length=2, depth=2, number_of_junctions=2)#self.initial_filter) #HarmonicSeries_V4(x)
        convnet_model = Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def HarmonicSeriesV5(self):
        shape_default = (self.input_shape[0], self.input_shape[1],self.number_input_channel)#, number_input_channel)
        x = layers.Input(shape=shape_default)

        filter_size =16
        #512x512
        x_ = utils.RotationThetaWeightLayer()([x,x])
        x_y = self.Conv2DBNSLU(x_, filters= 16, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch5x5 =  self.Conv2DBNSLU(x_,filters= 16, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch7x7 =  self.Conv2DBNSLU(x_,filters= 16, kernel_size=(7, 7), strides=2, activation='relu', padding='same')
        branch16x16 =  self.Conv2DBNSLU(x_,filters= 16, kernel_size=(16, 16), strides=2, activation='relu', padding='same')
        
        #x_y = layers.Dropout(0.2)(x_y)
        x_y = layers.concatenate(
            [branch5x5, branch7x7, branch16x16, x_y],
            axis=3)
        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #128x128x64
        x_y_0 = layers.Conv2D(32, kernel_size=(5,5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(32,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(32,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        #64x64x(96*3)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0],
            axis=3)
        #x_y = layers.Dropout(0.2)(x_y)
        #32x32x128
        x_y =  layers.Conv2D(128,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
    
        #16x16x(128x3)
        x_y_0_0 =  layers.Conv2D(256,kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(256,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(256,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0],
            axis=3)
        #x_y = layers.Dropout(0.2)(x_y)
        #8x8x(128x3)
        x_y =  layers.Conv2D(512,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #4x4x
        x_y_0_0_0 = layers.Conv2D(1024, kernel_size=(3, 3),strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 = layers.Conv2D(1024, kernel_size=(1, 3),strides=2, activation='relu', padding='same')(x_y)
        branch1xdim = layers.Conv2D(1024, kernel_size=(3, 1),strides=2, activation='relu', padding='same')(x_y)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0_0],
            axis=3)
        #x_y = layers.Dropout(0.2)(x_y)
        
        y = layers.Conv2D(2048, kernel_size=(1,1),strides=(1,1), activation='relu', padding='same')(x_y)
        
        #8x8 -->
        y = layers.GlobalAveragePooling2D()(y)
        y = layers.Dense(2048, activation='relu')(y)
        #y = layers.Dropout(0.1)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation, kernel_regularizer=regularizers.l2(0.0001))(y)

        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    
    def _conv_block(self, x, initial_filter, reduction_channel_ratio=0.5, kernel_regularizer=None, seed=0):
        x_v_0 = layers.Conv2D(initial_filter, (1,1),kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed))(x)
        
        x_v_1_0 = layers.Conv2D(int(round(initial_filter*1.5)), (1,1), kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.he_uniform(seed=seed+1))(x)
        x_v_1_1 = layers.Conv2D(initial_filter, (1,3),padding='same')(x_v_1_0)
        x_v_1_2 = layers.Conv2D(initial_filter, (3,1),padding='same')(x_v_1_0)
        
        x_v_2 = layers.Conv2D(int(round(initial_filter*1.5)), (1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+2))(x)
        x_v_2 = layers.Conv2D(int(round(initial_filter*1.75)), (1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+3))(x_v_2)
        x_v_2 = layers.Conv2D(int(round(initial_filter*2)), (3,1),padding='same',kernel_initializer=initializers.glorot_uniform(seed=seed+4))(x_v_2)
        x_v_2_0 = layers.Conv2D(initial_filter, (3,1),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+5))(x_v_2)
        x_v_2_1 = layers.Conv2D(initial_filter, (1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+6))(x_v_2)
        
        x_v_3 = layers.AveragePooling2D((2, 2), strides=(1,1),padding='same')(x)
        x_v_3 = layers.Conv2D(initial_filter, (1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+7))(x_v_3)
        
        x_y = layers.Concatenate()([x_v_0, x_v_1_1,x_v_1_2, x_v_2_0,x_v_2_1, x_v_3])
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        shape_c = x_y.shape.as_list()[-1]
        x_y = layers.Conv2D(int(round(reduction_channel_ratio*float(shape_c))), (1,1), strides=(1,1), padding='same', kernel_initializer=initializers.he_uniform(seed=seed+8))(x_y)
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        return x_y

    def _conv3d_block(self, x, initial_filter, reduction_channel_ratio=0.5, kernel_regularizer=None, seed=0):
        x_v_0 = layers.Conv3D(initial_filter, (1,1,1),kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed))(x)
        
        x_v_1_0 = layers.Conv3D(int(round(initial_filter*1.5)), (1,1,1), kernel_regularizer=kernel_regularizer, padding='same', kernel_initializer=initializers.he_uniform(seed=seed+1))(x)
        x_v_1_1 = layers.Conv3D(initial_filter, (2,1,3),padding='same')(x_v_1_0)
        x_v_1_2 = layers.Conv3D(initial_filter, (2,3,1),padding='same')(x_v_1_0)
        
        x_v_2 = layers.Conv3D(int(round(initial_filter*1.5)), (1,1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+2))(x)
        x_v_2 = layers.Conv3D(int(round(initial_filter*1.75)), (2,1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+3))(x_v_2)
        x_v_2 = layers.Conv3D(int(round(initial_filter*2)), (2,3,1),padding='same',kernel_initializer=initializers.glorot_uniform(seed=seed+4))(x_v_2)
        x_v_2_0 = layers.Conv3D(initial_filter, (2,3,1),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+5))(x_v_2)
        x_v_2_1 = layers.Conv3D(initial_filter, (2,1,3),padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+6))(x_v_2)
        
        x_v_3 = layers.AveragePooling3D((1,2, 2), strides=(1,1,1),padding='same')(x)
        x_v_3 = layers.Conv3D(initial_filter, (1,1,1), kernel_regularizer=kernel_regularizer,padding='same', kernel_initializer=initializers.glorot_uniform(seed=seed+7))(x_v_3)
        
        x_y = layers.Concatenate()([x_v_0, x_v_1_1,x_v_1_2, x_v_2_0,x_v_2_1, x_v_3])
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        shape_c = x_y.shape.as_list()[-1]
        x_y = layers.Conv3D(int(round(reduction_channel_ratio*float(shape_c))), (1,1,1), strides=(1,1,1), padding='same', kernel_initializer=initializers.he_uniform(seed=seed+8))(x_y)
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        return x_y

    def Spider_Node(self, x_input, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node = []
        x = x_input
        for i in range(depth):
            x = self._conv_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            node.append(x)
            x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node

    def Spider_3D_Net(self, x, initial_filter=32, compression=0.5, length=5, depth=7, center_node_id=0, kernel_regularizer=None,random_junctions=True, number_of_junctions=5, junction_only_the_last_layers=False):
        nodes = []

        #Generate nodes
        for i in range(length):
            nodes.append(self.Spider_3D_Node(x, initial_filter, compression,depth, kernel_regularizer, counter=i))

        #Generate Connection between Nodes
        
        #Generate the junctions between nodes
        connected_layers = []
        con_A, con_B, _ = self.Connect_Nodes(nodes)
        with open('./junction.txt', 'w') as f:
            for (conv_A, conv_B) in zip(con_A,con_B):
                connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
                f.write("%s %s\n" % (conv_A, conv_B))
        
        iteration_data = self.GetIterationConnectionKeys(number_of_nodes=length, depth=depth, random_junctions=random_junctions,number_of_junctions=number_of_junctions)
        
        #Update the node structure:
        itm_nbr = iteration_data.shape[0]
        for i in range(itm_nbr):
            row = iteration_data[i]
            node_id, level_id, junc_id = row[0], row[1], row[2]
            if junction_only_the_last_layers:
                nodes[node_id][level_id] = layers.Concatenate()([connected_layers[junc_id],nodes[node_id][level_id]])
            else:
                nodes[node_id][level_id] = connected_layers[junc_id]

        if junction_only_the_last_layers:
            y = connected_layers[-(length-1):]
            y = layers.concatenate(y)
        elif random_junctions==False:
            second_nodes = []
            for i in range(length):
                if i != center_node_id:
                    second_nodes.append(self.Spider_3D_Node_w_Junction(x, nodes[i], initial_filter,compression,depth, kernel_regularizer, counter=i))
                else:
                    second_nodes.append(nodes[i])
            last_connection = []
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            
            y = layers.concatenate(last_connection)
        elif random_junctions:
            second_nodes = []
            node_id_junctions = iteration_data[:,0].tolist()
            print(node_id_junctions)
            for i in range(length):
                if i == center_node_id:
                    second_nodes.append(nodes[i])
                else:
                    if i in node_id_junctions:
                        junction_levels = iteration_data[iteration_data[:,0]==i][:,1].tolist()
                    else:
                        junction_levels = None
                    second_nodes.append(self.Spider_3D_Node_w_Junction_list(x, nodes[i], initial_filter, compression, depth, junction_levels,kernel_regularizer, counter=i))
            last_connection = []
            print(len(second_nodes))
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            
            y = layers.concatenate(last_connection)
        else:
            raise NameError('Please specify the arguments, junction_only_the_last_layers, random_junctions')

        #FC
        y = layers.GlobalMaxPooling3D()(y)
        dense_shape = y.shape.as_list()[-1]
        #y = layers.Reshape(target_shape=[16,-1, 48])(y)
        #y = layers.Conv2D(1,(1,1296), activation=utils.softmax)(y)
        #y = layers.Flatten()(y)
        #dense_shape = 16
        y = layers.Dense(dense_shape, activation= 'softmax')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        return y

    def Spider_3D_Node(self, x_input, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node = []
        x = x_input
        for i in range(depth):
            x = self._conv3d_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            node.append(x)
            x = layers.AveragePooling3D((1, 2,2), strides=(1, 2,2))(x) 
        return node

    def Spider_3D_Node_w_Junction_list(self, x_input, node, filter,compression=0.5, depth=5, junction_list=None, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node_tmp = []
        x = x_input
        for i in range(depth):
            if junction_list is None:
                pass
            elif i in junction_list:
                x = layers.concatenate([x,node[i]])
            else:
                pass
            x = self._conv3d_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            
            node_tmp.append(x)
            x = layers.AveragePooling3D((1,2, 2), strides=(1,2, 2))(x) 
        return node_tmp

    def Spider_3D_Node_w_Junction(self, x_input, node, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node_tmp = []
        x = x_input
        for i in range(depth):
            x = self._conv3d_block(layers.concatenate([x,node[i]]), filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            node_tmp.append(x)
            x = layers.AveragePooling3D((1,2, 2), strides=(1,2, 2))(x) 
        return node_tmp

    def Spider_Node_w_Junction(self, x_input, node, filter,compression=0.5, depth=5, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node_tmp = []
        x = x_input
        for i in range(depth):
            x = self._conv_block(layers.concatenate([x,node[i]]), filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            node_tmp.append(x)
            x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node_tmp

    def Spider_Node_w_Junction_list(self, x_input, node, filter,compression=0.5, depth=5, junction_list=None, kernel_regularizer=regularizers.l2(0.00001), counter=0):
        node_tmp = []
        x = x_input
        for i in range(depth):
            if junction_list is None:
                pass
            elif i in junction_list:
                x = layers.concatenate([x,node[i]])
            else:
                pass
            x = self._conv_block(x, filter*(i+1)+2, reduction_channel_ratio=compression, kernel_regularizer=kernel_regularizer, seed=(i+counter))
            
            node_tmp.append(x)
            x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x) 
        return node_tmp

    def Connect_Nodes(self, nodes, center_node_id=0):
        number_of_nodes = len(nodes)
        length_of_node = len(nodes[0])
        connection_A =[]
        connection_B = []
        type_of_junction =[]
        for i in range(length_of_node):
            for j in range(number_of_nodes):
                if j != center_node_id:
                    connection_A.append(nodes[center_node_id][i])
                    connection_B.append(nodes[j][i])
                    type_of_junction.append(layers.add)
        return connection_A, connection_B, type_of_junction

    def Connect_Different_Nodes(self, nodes, junctions, iteration_data):
        itm_nbr = iteration_data.shape[0]
        for i in range(itm_nbr):
            row = iteration_data[i]
            node_id, level_id, junc_id = row[0], row[1], row[2]
            print(junctions[junc_id])
            print(nodes[node_id][level_id])
            nodes[node_id][level_id] = layers.Concatenate()([junctions[junc_id],nodes[node_id][level_id]])
        return nodes

    def GetIterationConnectionKeys(self,number_of_nodes, depth, random_junctions=True, number_of_junctions=12):
        nodes = list(range(1, number_of_nodes))
        depth_lst = list(range(0,depth))
        counter = list(range(0,(number_of_nodes-1)*depth))
        
        import numpy as np
        nodes_ = np.array(nodes*depth)
        depth_ = np.array(depth_lst*(number_of_nodes-1))
        counter = np.array(counter)
        np_ = np.zeros((((number_of_nodes-1)*depth),3), dtype=np.int)

        np_[...,0] = nodes_
        np_[...,1] = depth_
        np_[...,2] = counter
        for i in range(depth):
            np_[i*(number_of_nodes-1):(i+1)*(number_of_nodes-1),1] = i
        if random_junctions and number_of_junctions>0:
            indexes = list(range(np_.shape[0]))
            random.seed=1
            indexes_selected = random.sample(indexes,number_of_junctions)
            random.seed=None
            np_ = np_[indexes_selected]
            print(np_.shape)
        return np_
    #length=6, depth=8
    
    def Spider_Net(self, x, initial_filter=32, compression=0.5, length=5, depth=7, center_node_id=0, kernel_regularizer=None,random_junctions=True, number_of_junctions=5, junction_only_the_last_layers=False):
        nodes = []

        #Generate nodes
        for i in range(length):
            nodes.append(self.Spider_Node(x, initial_filter, compression,depth, kernel_regularizer, counter=i))

        #Generate Connection between Nodes
        
        #Generate the junctions between nodes
        connected_layers = []
        con_A, con_B, _ = self.Connect_Nodes(nodes)
        with open('./junction.txt', 'w') as f:
            for (conv_A, conv_B) in zip(con_A,con_B):
                connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
                f.write("%s %s\n" % (conv_A, conv_B))
        '''
        if os.path.exists('./junction.txt'):
            with open('./junction.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    conv_A, conv_B = line.split(' ')
                    connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
        else:
            con_A, con_B, _ = self.Connect_Nodes(nodes)
            with open('./junction.txt', 'w') as f:
                for (conv_A, conv_B) in zip(con_A,con_B):
                    connected_layers.append(utils.JunctionWeightLayer()([conv_A,conv_B]))
                    f.write("%s %s\n" % (conv_A, conv_B))
        '''
        iteration_data = self.GetIterationConnectionKeys(number_of_nodes=length, depth=depth, random_junctions=random_junctions,number_of_junctions=number_of_junctions)
        
        #Update the node structure:
        itm_nbr = iteration_data.shape[0]
        for i in range(itm_nbr):
            row = iteration_data[i]
            node_id, level_id, junc_id = row[0], row[1], row[2]
            if junction_only_the_last_layers:
                nodes[node_id][level_id] = layers.Concatenate()([connected_layers[junc_id],nodes[node_id][level_id]])
            else:
                nodes[node_id][level_id] = connected_layers[junc_id]

        if junction_only_the_last_layers:
            y = connected_layers[-(length-1):]
            y = layers.concatenate(y)
        elif random_junctions==False:
            second_nodes = []
            for i in range(length):
                if i != center_node_id:
                    second_nodes.append(self.Spider_Node_w_Junction(x, nodes[i], initial_filter,compression,depth, kernel_regularizer, counter=i))
                else:
                    second_nodes.append(nodes[i])
            last_connection = []
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            
            y = layers.concatenate(last_connection)
        elif random_junctions:
            second_nodes = []
            node_id_junctions = iteration_data[:,0].tolist()
            print(node_id_junctions)
            for i in range(length):
                if i == center_node_id:
                    second_nodes.append(nodes[i])
                else:
                    if i in node_id_junctions:
                        junction_levels = iteration_data[iteration_data[:,0]==i][:,1].tolist()
                    else:
                        junction_levels = None
                    second_nodes.append(self.Spider_Node_w_Junction_list(x, nodes[i], initial_filter, compression, depth, junction_levels,kernel_regularizer, counter=i))
            last_connection = []
            print(len(second_nodes))
            for i in range(length):
                last_connection.append(second_nodes[i][-1])
            
            y = layers.concatenate(last_connection)
        else:
            raise NameError('Please specify the arguments, junction_only_the_last_layers, random_junctions')

        #FC
        y = layers.GlobalMaxPooling3D()(y)
        dense_shape = y.shape.as_list()[-1]
        #dense_shape = 1024
        y = layers.Dense(dense_shape, activation= 'relu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        return y

    def __block_v1(self, x, initial_filter=16, kernel_regularizer=None):
        x_v_0 = layers.Conv2D(96, (1,1),kernel_regularizer=kernel_regularizer)(x)

        x_v_1 = layers.Conv2D(64, (1,1),kernel_regularizer=kernel_regularizer)(x)
        x_v_1 = layers.Conv2D(96, (3,3))(x_v_1)

        x_v_2 = layers.Conv2D(64, (1,1),kernel_regularizer=kernel_regularizer)(x)
        x_v_2 = layers.Conv2D(96, (3,3))(x_v_2)
        x_v_2 = layers.Conv2D(96, (3,3))(x_v_2)

        x_v_3 = layers.AveragePooling2D((2, 2),strides=(1,1), padding='same')(x)
        x_v_3 = layers.Conv2D(96, (1,1),kernel_regularizer=kernel_regularizer)(x)

        x_y = layers.Concatenate([x_v_0, x_v_1, x_v_2, x_v_3])
        x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
        x_v = layers.Activation('relu')(x_y)
        return x_y

    def __block_v2(self, x, kernel_regularizer=None):
        x_v_0 = layers.Conv2D(256, (1,1),kernel_regularizer=kernel_regularizer)(x)

        x_v_1_0 = layers.Conv2D(386, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_1_1 = layers.Conv2D(256, (1,3))(x_v_1_0)
        x_v_1_2 = layers.Conv2D(256, (3,1))(x_v_1_0)

        x_v_2 = layers.Conv2D(386, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_2 = layers.Conv2D(448, (1,3))(x_v_2)
        x_v_2 = layers.Conv2D(512, (3,1))(x_v_2)
        x_v_2_0 = layers.Conv2D(256, (3,1))(x_v_2)
        x_v_2_1 = layers.Conv2D(256, (1,3))(x_v_2)

        x_v_3 = layers.AveragePooling2D((2, 2), strides=(1,1) ,padding='same')(x)
        x_v_3 = layers.Conv2D(256, (1,1), kernel_regularizer=kernel_regularizer)(x)

        x_y = layers.Concatenate([x_v_0, x_v_1_1,x_v_1_0, x_v_2_0,x_v_2_1, x_v_3])
        x_y = layers.BatchNormalization(epsilon=1.1e-5)(x_y)
        x_v = layers.Activation('relu')(x_y)
        return x_y

    def __block_7x1(self, x, kernel_regularizer=None):
        x_v_0 = layers.Conv2D(384, (1,1), kernel_regularizer=kernel_regularizer)(x)

        x_v_1 = layers.Conv2D(196, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_1 = layers.Conv2D(224, (1,7))(x_v_1)
        x_v_1 = layers.Conv2D(256, (7,1))(x_v_1)

        x_v_2 = layers.Conv2D(192, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_2 = layers.Conv2D(192, (1,7))(x_v_2)
        x_v_2 = layers.Conv2D(224, (7,1))(x_v_2)
        x_v_2 = layers.Conv2D(224, (7,1))(x_v_2)
        x_v_2 = layers.Conv2D(256, (7,1))(x_v_2)

        x_v_3 = layers.AveragePooling2D((2, 2),strides=(1,1), padding='same')(x)
        x_v_3 = layers.Conv2D(128, (1,1), kernel_regularizer=kernel_regularizer)(x)

        x_y = layers.Concatenate([x_v_0, x_v_1, x_v_2, x_v_3])
        x_y = layers.BatchNormalization(epsilon=1.1e-5, scale=False)(x_y)
        x_v = layers.Activation('relu')(x_y)
        return x_y
    
    def __block_5x1(self, x, kernel_regularizer=None):
        x_v_0 = layers.Conv2D(384, (1,1),kernel_regularizer=kernel_regularizer)(x)

        x_v_1 = layers.Conv2D(196, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_1 = layers.Conv2D(224, (1,5))(x_v_1)
        x_v_1 = layers.Conv2D(256, (5,1))(x_v_1)

        x_v_2 = layers.Conv2D(192, (1,1), kernel_regularizer=kernel_regularizer)(x)
        x_v_2 = layers.Conv2D(192, (1,5))(x_v_2)
        x_v_2 = layers.Conv2D(224, (5,1))(x_v_2)
        x_v_2 = layers.Conv2D(224, (5,1))(x_v_2)
        x_v_2 = layers.Conv2D(256, (5,1))(x_v_2)

        x_v_3 = layers.AveragePooling2D((2, 2),strides=(1,1), padding='same')(x)
        x_v_3 = layers.Conv2D(128, (1,1), kernel_regularizer=kernel_regularizer)(x)

        x_y = layers.Concatenate([x_v_0, x_v_1, x_v_2, x_v_3])
        x_y = layers.BatchNormalization(epsilon=1.1e-5)(x_y)
        x_v = layers.Activation('relu')(x_y)
        return x_y

    def __reduction_block(self, nb_filter, compression, x, kernel_regularizer=None):
        x = layers.BatchNormalization(axis=-1, epsilon=1.1e-5, scale=False)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                kernel_regularizer=kernel_regularizer)(x)
        x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    def ModelType_Gls4_3_Type10_v2(self):
        #print('ModelType_Gls4_3_Type_10')
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)

        x_y = layers.Lambda(lambda x: x/255.0)(x)
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        x_y = utils.RotationThetaWeightLayer()([x_y,x_y])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #
        x_y = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y)
        
        
        #512x512
        for i in range(2):
            x_y_01 = layers.Conv2D(filters= 8, kernel_size=(7, 7), strides=2, padding='same')(x_y)
            x_y_02 = layers.Conv2D(filters= 8, kernel_size=(9, 9), strides=2, padding='same')(x_y)
            x_y_03 = layers.Conv2D(filters= 8, kernel_size=(5, 5), strides=2, padding='same')(x_y)
            x_y_04 = layers.Conv2D(filters= 8, kernel_size=(3, 3), strides=2, padding='same')(x_y)
            x_y = layers.concatenate([x_y_01, x_y_02, x_y_03, x_y_04])
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        x_y_0 = layers.Conv2D(filters= 32, kernel_size=(1, 1), strides=1, padding='same')(x_y)
        for i in range(2):
            x_y_01 = layers.Conv2D(filters= 16, kernel_size=(1, 5), strides=1, padding='same')(x_y)
            x_y_02 = layers.Conv2D(filters= 16, kernel_size=(5, 1), strides=1, padding='same')(x_y)
            x_y_03 = layers.Conv2D(filters= 16, kernel_size=(5, 5), strides=1, padding='same')(x_y)
            x_y_04 = layers.Conv2D(filters= 16, kernel_size=(3, 3), strides=1, padding='same')(x_y)
            x_y = layers.concatenate([x_y_01, x_y_02, x_y_03, x_y_04])
            x_y = layers.Conv2D(filters= 32, kernel_size=(1, 1), strides=1, padding='same')(x_y)
        
        x_y = layers.add([x_y_0,x_y])
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y_1 = layers.Activation('relu')(x_y)

        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y_1)
        x_y_1 = layers.Conv2D(filters= 64, kernel_size=(1, 1), strides=1, padding='same')(x_y)
        for i in range(3):
            x_y_01 = layers.Conv2D(filters= 32, kernel_size=(1, 5), strides=1, padding='same')(x_y)
            x_y_02 = layers.Conv2D(filters= 32, kernel_size=(5, 1), strides=1, padding='same')(x_y)
            x_y_03 = layers.Conv2D(filters= 32, kernel_size=(5, 5), strides=1, padding='same')(x_y)
            x_y_04 = layers.Conv2D(filters= 32, kernel_size=(3, 3), strides=1, padding='same')(x_y)
            
            x_y = layers.concatenate([x_y_01, x_y_02, x_y_03, x_y_04])
            x_y = layers.Conv2D(filters= 64, kernel_size=(1, 1), strides=1, padding='same')(x_y)

        x_y = layers.add([x_y_1,x_y])
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        for j in [128,256,512]:
            filter_nr_x = x_y.shape.as_list()[-1]
            
            x_y_0 = layers.Conv2D(filters= int(round(j*0.8)), kernel_size=(1, 1), strides=1, padding='same')(x_y)
            x_y_0 = layers.Conv2D(filters= filter_nr_x, kernel_size=(5, 5), strides=1, padding='same')(x_y_0)
            x_y = layers.add([x_y_0,x_y])
            x_y = layers.BatchNormalization()(x_y)
            x_y = layers.Activation('relu')(x_y)
            
            x_y_1 = layers.Conv2D(filters= int(round(j*0.8)), kernel_size=(1, 1), strides=1, padding='same')(x_y)

            x_y_1 = layers.Conv2D(filters= j, kernel_size=(3, 1), strides=1, padding='same')(x_y_1)
            x_y_1 = layers.Conv2D(filters= j, kernel_size=(1, 3), strides=1, padding='same')(x_y_1)
            
            x_y_2 = layers.Conv2D(filters= j, kernel_size=(1, 3), strides=1, padding='same')(x_y)
            x_y_2 = layers.Conv2D(filters= j, kernel_size=(3, 1), strides=1, padding='same')(x_y_2)
            
            x_y_3 = layers.Conv2D(filters= j, kernel_size=(3, 3), strides=1, padding='same')(x_y)
            x_y_3 = layers.Conv2D(filters= j, kernel_size=(3, 3), strides=1, padding='same')(x_y_3)
            
            x_y = layers.add([x_y_1,x_y_3, x_y_2])

            #x_y = layers.concatenate([x_y_0, x_y_1,x_y_2, x_y_3])
            x_y = layers.BatchNormalization(scale=False)(x_y)
            x_y = layers.Activation('relu')(x_y)
            
            if j != 512:
                x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y = layers.Conv2D(filters= 128, kernel_size=1, strides=1, padding='same')(x_y)
        x_y = layers.BatchNormalization(scale=False)(x_y)
        x_y = layers.Activation('relu')(x_y)
        y = layers.Flatten()(x_y)
        y = layers.Dense(2048, activation='relu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model

    def HarmonicSeries_V4(self,x):
        #shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        #x = layers.Input(shape=shape_default)

        filter_size =16
        #512x512
        #x_y = layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(x)
        #512x512
        x_y = self.Conv2DBNSLU(x, filters= 16, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x, filters= 16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        branch3x3 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(3, 3), strides=2,activation=None,  padding='same', BN=True)
        branch3x3 =  self.Conv2DBNSLU(branch3x3,filters= 16, kernel_size=(3, 1), strides=1,activation=None,  padding='same', BN=True)
        branch3x3 =  self.Conv2DBNSLU(branch3x3,filters= 16, kernel_size=(1, 3), strides=1,activation=None,  padding='same',BN=True)
        
        branch5x5 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(5, 5), strides=2,activation=None,  padding='same',BN=True)
        branch5x5 =  self.Conv2DBNSLU(branch5x5,filters= 16, kernel_size=(5, 1), strides=1,activation=None,  padding='same',BN=True)
        branch5x5 =  self.Conv2DBNSLU(branch5x5,filters= 16, kernel_size=(1, 5), strides=1,activation=None,  padding='same',BN=True)
        
        x_y = layers.concatenate(
            [branch5x5, x_y, branch3x3],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        #256x256x64
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        #128x128x64
        pool_4x4= layers.Conv2D(32, kernel_size=(3, 3),strides=1, padding='same')(x_y)
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        x_y_0 = layers.Conv2D(32, kernel_size=(3,3), strides=2, padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(32,kernel_size=(3, 1), strides=2, padding='same')(x_y)
        branch1xdim =  layers.Conv2D(32,kernel_size=(1, 3), strides=2, padding='same')(x_y)
        #64x64x(96*3)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #32x32x128
        x_y =  layers.Conv2D(64,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)#128
        x_y =  layers.Conv2D(256,kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x_y) # 512

        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
    
        #16x16x(128x3) 
        pool_4x4= layers.Conv2D(128, kernel_size=(3, 3),strides=1, activation=None, padding='same')(x_y)  #256
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        x_y_0_0 =  layers.Conv2D(128,kernel_size=(3, 3), strides=2, activation=None, padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(128,kernel_size=(3, 1), strides=2, activation=None, padding='same')(x_y)
        branch1xdim =  layers.Conv2D(128,kernel_size=(1, 3), strides=2, activation=None, padding='same')(x_y)
        
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #8x8x(128x3)
        x_y =  layers.Conv2D(256,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y) # 512
        x_y =  layers.Conv2D(256,kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x_y) # 512

        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #4x4x 
        x_y_0_0_0 = layers.Conv2D(512, kernel_size=(3, 3),strides=2, activation=None, padding='same')(x_y) #1024
        pool_4x4= layers.Conv2D(512, kernel_size=(3, 3),strides=1, activation=None, padding='same')(x_y)
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        branchdimx1 = layers.Conv2D(512, kernel_size=(1, 3),strides=2, activation=None, padding='same')(x_y)
        branch1xdim = layers.Conv2D(512, kernel_size=(3, 1),strides=2, activation=None, padding='same')(x_y)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0_0_0],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #y = layers.Conv2D(1024, kernel_size=(1,1),strides=(1,1), activation='relu', padding='same')(x_y) #2048
        
        #8x8 -->
        y = layers.GlobalMaxPool2D()(x_y)
        y = layers.Dense(2048, activation='relu')(y) #1024
        y = layers.Dense(self.n_class, activation=self.final_activation,name='class')(y)
        
        #y = layers.Reshape((4,4),name='class')(y)
        #convnet_model = models.Model(inputs=x, outputs=y)
        return y#convnet_model
    
    def HarmonicSeries_V4_v(self,x):
        #shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        #x = layers.Input(shape=shape_default)

        filter_size =16
        #512x512
        #x_y = layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(x)
        #512x512
        x_y = self.Conv2DBNSLU(x, filters= 16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x, filters= 16, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        branch5x5 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(5, 5), strides=2,activation=None, padding='same')
        branch7x7 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(7, 7), strides=2,activation=None,  padding='same')
        branch3x3 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(3, 3), strides=2,activation=None,  padding='same')
        
        x_y = layers.concatenate(
            [branch5x5, branch7x7, x_y, branch3x3],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        #256x256x64
        x_y = layers.Activation('relu')(x)
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #128x128x64
        pool_4x4= layers.Conv2D(32, kernel_size=(3, 3),strides=1, padding='same')(x_y)
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        x_y_0 = layers.Conv2D(32, kernel_size=(5,5), strides=2, padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(32,kernel_size=(5, 1), strides=2, padding='same')(x_y)
        branch1xdim =  layers.Conv2D(32,kernel_size=(1, 5), strides=2, padding='same')(x_y)
        #64x64x(96*3)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #32x32x128
        x_y =  layers.Conv2D(64,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)#128
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
    
        #16x16x(128x3) 
        pool_4x4= layers.Conv2D(128, kernel_size=(3, 3),strides=1, activation=None, padding='same')(x_y)  #256
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        x_y_0_0 =  layers.Conv2D(128,kernel_size=(5, 5), strides=2, activation=None, padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(128,kernel_size=(5, 3), strides=2, activation=None, padding='same')(x_y)
        branch1xdim =  layers.Conv2D(128,kernel_size=(3, 5), strides=2, activation=None, padding='same')(x_y)
        
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #8x8x(128x3)
        x_y =  layers.Conv2D(256,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y) # 512
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #4x4x 
        x_y_0_0_0 = layers.Conv2D(512, kernel_size=(3, 3),strides=2, activation=None, padding='same')(x_y) #1024
        pool_4x4= layers.Conv2D(512, kernel_size=(3, 3),strides=1, activation=None, padding='same')(x_y)
        pool_4x4 = layers.MaxPooling2D((2, 2), padding='same')(pool_4x4)
        branchdimx1 = layers.Conv2D(512, kernel_size=(1, 3),strides=2, activation=None, padding='same')(x_y)
        branch1xdim = layers.Conv2D(512, kernel_size=(3, 1),strides=2, activation=None, padding='same')(x_y)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, pool_4x4, x_y_0_0_0],
            axis=3)
        x_y = layers.BatchNormalization()(x_y)
        x_y = layers.Activation('relu')(x_y)
        #y = layers.Conv2D(1024, kernel_size=(1,1),strides=(1,1), activation='relu', padding='same')(x_y) #2048
        
        #8x8 -->
        y = layers.GlobalMaxPool2D()(x_y)
        y = layers.Dense(2048, activation='relu')(y) #1024
        y = layers.Dense(self.n_class, activation=self.final_activation,name='class')(y)
        
        #y = layers.Reshape((4,4),name='class')(y)
        #convnet_model = models.Model(inputs=x, outputs=y)
        return y#convnet_model
    
    def HarmonicSeries_V4_X(self):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)

        filter_size =16
        #512x512
        #x_y = layers.Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same')(x)
        #512x512
        x_y = self.Conv2DBNSLU(x, filters= 16, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch5x5 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch7x7 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(7, 7), strides=2, activation='relu', padding='same')
        branch16x16 =  self.Conv2DBNSLU(x,filters= 16, kernel_size=(16, 16), strides=2, activation='relu', padding='same')
        
        x_y = layers.Dropout(0.2)(x_y)
        x_y = layers.concatenate(
            [branch5x5, branch7x7, branch16x16, x_y],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #128x128x64
        x_y_0 = layers.Conv2D(32, kernel_size=(5,5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(32,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(32,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        #64x64x(96*3)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.Dropout(0.2)(x_y)
        #32x32x128
        x_y =  layers.Conv2D(128,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
    
        #16x16x(128x3)
        x_y_0_0 =  layers.Conv2D(256,kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(256,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(256,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0],
            #[branch1x1, branch5x5, branch7x7, branch3x3dbl, branch1xdim, branchdimx1, branch_pool],
            axis=3)
        x_y = layers.Dropout(0.2)(x_y)
        #8x8x(128x3)
        x_y =  layers.Conv2D(512,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #4x4x
        x_y_0_0_0 = layers.Conv2D(1024, kernel_size=(3, 3),strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 = layers.Conv2D(1024, kernel_size=(1, 3),strides=2, activation='relu', padding='same')(x_y)
        branch1xdim = layers.Conv2D(1024, kernel_size=(3, 1),strides=2, activation='relu', padding='same')(x_y)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0_0],
            axis=3)
        x_y = layers.Dropout(0.2)(x_y)
        y = layers.Conv2D(2048, kernel_size=(1,1),strides=(1,1), activation='relu', padding='same')(x_y)
        
        #8x8 -->
        y = layers.GlobalAveragePooling2D()(y)
        y = layers.Dense(2048, activation='relu')(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation, name='class')(y)

        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    
    def Inception(self, freeze=True):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        #input_x = layers.Input(shape_default)
        base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=shape_default, pooling='max', classes=None)
        if freeze==True:
            for layer in base_model.layers[:249]:
                layer.trainable = True
            for layer in base_model.layers[249:]:
                layer.trainable = False
        x = base_model.output
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        y = layers.Dense(self.n_class, activation=self.final_activation)(x)
        convnet_model = models.Model(inputs=base_model.input, outputs=y)
        return convnet_model

    def HarmonicSeries(self):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)

        filter_size =self.initial_filter

        x_y = self.Conv2DBNSLU(x, filters= filter_size, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch5x5 =  self.Conv2DBNSLU(x,filters= filter_size, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        branch7x7 =  self.Conv2DBNSLU(x,filters= filter_size, kernel_size=(7, 7), strides=2, activation='relu', padding='same')
        branch16x16 =  self.Conv2DBNSLU(x,filters= filter_size, kernel_size=(16, 16), strides=2, activation='relu', padding='same')
        
        x_y = layers.Dropout(0.1)(x_y)
        x_y = layers.concatenate(
            [branch5x5, branch7x7, branch16x16, x_y],
            axis=3)

        #256x256x64
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #128x128x64
        x_y_0 = layers.Conv2D(filter_size*2, kernel_size=(5,5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(filter_size*2,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(filter_size*2,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        #64x64x(96*3)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0],
            axis=3)
        x_y = layers.Dropout(0.1)(x_y)
        #32x32x128
        
        x_y =  layers.Conv2D(filter_size*8,kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
    
        #16x16x(128x3)
        x_y_0_0 =  layers.Conv2D(filter_size*16,kernel_size=(5, 5), strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 =  layers.Conv2D(filter_size*16,kernel_size=(5, 3), strides=2, activation='relu', padding='same')(x_y)
        branch1xdim =  layers.Conv2D(filter_size*16,kernel_size=(3, 5), strides=2, activation='relu', padding='same')(x_y)
        
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0],
            axis=3)
        x_y = layers.Dropout(0.1)(x_y)
        #8x8x(128x3)
        x_y =  layers.Conv2D(filter_size*32,kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x_y)
        x_y =  layers.Conv2D(filter_size*32,kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x_y)
        x_y =  layers.Conv2D(filter_size*32,kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        #4x4x
        x_y_0_0_0 = layers.Conv2D(filter_size*64, kernel_size=(3, 3),strides=2, activation='relu', padding='same')(x_y)
        branchdimx1 = layers.Conv2D(filter_size*64, kernel_size=(1, 3),strides=2, activation='relu', padding='same')(x_y)
        branch1xdim = layers.Conv2D(filter_size*64, kernel_size=(3, 1),strides=2, activation='relu', padding='same')(x_y)
        x_y = layers.concatenate(
            [branchdimx1, branch1xdim, x_y_0_0_0],
            axis=3)
        x_y = layers.Dropout(0.1)(x_y)
        y = layers.Conv2D(filter_size*128, kernel_size=(1,1),strides=(1,1), activation='relu', padding='same')(x_y)
        
        y = layers.GlobalAveragePooling2D()(y)
        y = layers.Dense(filter_size*128, activation='relu')(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)

        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    def Simple_layers3(self):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        input_x = layers.Input(shape=shape_default)
        #1. 512 -> 128
        activation_func = 'relu'
        x = self.Conv2DBNSLU(input_x, filters=32, kernel_size=(5, 5), strides=2, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=32, kernel_size=(5, 5), strides=2, activation=activation_func, padding='same')

        x = layers.Dropout(0.15)(x)
        x = layers.MaxPooling2D()(x)
        #2. 128 -> 32
        x = self.Conv2DBNSLU(x, filters=64, kernel_size=(3, 3), strides=2, activation=activation_func, padding='valid')
        x = self.Conv2DBNSLU(x, filters=64, kernel_size=(3, 3), strides=1, activation=activation_func, padding='valid')
        x = layers.Dropout(0.15)(x)
        x = layers.MaxPooling2D()(x)
        #3. 32 --> 16
        x = self.Conv2DBNSLU(x, filters=128, kernel_size=(3, 3), strides=1, activation=activation_func, padding='valid')
        x = self.Conv2DBNSLU(x, filters=128, kernel_size=(3, 3), strides=1, activation=activation_func, padding='valid')
        x = self.Conv2DBNSLU(x, filters=128, kernel_size=(3, 3), strides=1, activation=activation_func, padding='valid')       
        x = layers.Dropout(0.15)(x)
        x = layers.MaxPooling2D()(x)
        #FC
        y = layers.Flatten()(x)
        
        y = layers.Dense(2048, activation=activation_func)(y)
        y = layers.Dropout(0.05)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        convnet_model = models.Model(inputs=input_x, outputs=y)
        return convnet_model
    def SimpleSeries(self):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        input_x = layers.Input(shape=shape_default)
        filter_size = self.initial_filter
        #1. 512 -> 128
        activation_func = LeakyReLU()
        x = self.Conv2DBNSLU(input_x, filters=filter_size, kernel_size=(5, 5), strides=2, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size, kernel_size=(5, 5), strides=1, activation=activation_func, padding='same')
        x = layers.MaxPooling2D()(x)
        #2. 128 -> 32
        x = self.Conv2DBNSLU(x, filters=filter_size*2, kernel_size=(5, 5), strides=2, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*2, kernel_size=(5, 5), strides=1, activation=activation_func, padding='same')
        x = layers.MaxPooling2D()(x)
        #3. 32 --> 16
        x = self.Conv2DBNSLU(x, filters=filter_size*8, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*8, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*8, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = layers.MaxPooling2D()(x)
        #4. 16 --> 8
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = layers.MaxPooling2D()(x)
        #5. 8 --> 4
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = layers.MaxPooling2D()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*64, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*64, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = self.Conv2DBNSLU(x, filters=filter_size*64, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        
        #FC
        y = layers.GlobalMaxPooling2D()(x)
        y = layers.Dense(filter_size*64, activation=activation_func)(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        convnet_model = models.Model(inputs=input_x, outputs=y)
        return convnet_model

    def SimpleSeries5v(self):
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        input_x = layers.Input(shape=shape_default)
        filter_size = 16 #self.initial_filter
        #1. 512 -> 128
        activation_func = None
        x = self.Conv2DBNSLU(x, filters=filter_size, kernel_size=(4, 5), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size, kernel_size=(5, 4), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)
        #2. 128 -> 32
        x = self.Conv2DBNSLU(x, filters=filter_size*2, kernel_size=(5, 3), strides=2, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*2, kernel_size=(3, 5), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = layers.MaxPooling2D()(x)
        #3. 32 --> 16
        x = self.Conv2DBNSLU(x, filters=filter_size*8, kernel_size=(3, 3), strides=2, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*8, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        #x = layers.SpatialDropout2D(0.2)(x)
        x = layers.MaxPooling2D()(x)
        #4. 16 --> 8
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*16, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        #x = layers.SpatialDropout2D(0.2)(x)
        x = layers.MaxPooling2D()(x)
        #5. 8 --> 4
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        x = self.Conv2DBNSLU(x, filters=filter_size*32, kernel_size=(3, 3), strides=1, activation=activation_func, padding='same')
        x = LeakyReLU()(x)
        #x = layers.SpatialDropout2D(0.2)(x)
        #x = layers.SpatialDropout2D(0.2)(x)
        #FC
        y = layers.GlobalMaxPooling2D()(x)
        y = layers.Dense(filter_size*32, activation=activation_func)(y)
        y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        convnet_model = models.Model(inputs=input_x, outputs=y)
        return convnet_model
        
    def ModelType_Gls4_3_Type_20(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y = x
        #512x512
        for filter_n in [16,32, 64]:
            x_y = self.Conv2DBNSLU(x_y, filters= filter_n, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= filter_n, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        counter=1
        for j in [128,256,512,512,1024]:
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            x_y = layers.Dropout(0.2)(x_y)
            #x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
            
            if counter<5:
                x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
            counter +=1

        y = layers.GlobalAveragePooling2D()(x_y)
        y = layers.Dense(1024, activation='selu')(y)
        y = layers.SpatialDropout1D(0.2)(y)
        #y = layers.Dense(1024, activation='selu')(y)
        if self.final_activation is None:
            y = layers.Dense(self.n_class)(y)
        else: 
            y = layers.Dense(self.n_class, activation=self.final_activation)(y)
        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    
    def ComptactModel_v2(self,input_x):
        #shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = input_x #layers.Input(shape=shape_default)
        #512x512
        #Preprocessing segement
        x_y = self.Conv2DBNSLU(x, filters= 4, kernel_size=(4, 5), strides=1, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x_y, filters= 4, kernel_size=(5, 4), strides=1, activation=None, padding='same')
        
        x_y_01 = self.Conv2DBNSLU(x, filters= 4, kernel_size=(5, 1), strides=1, activation='relu', padding='same')
        x_y_01 = self.Conv2DBNSLU(x_y_01, filters= 4, kernel_size=(1, 5), strides=1, activation=None, padding='same')
        
        x_y_02 = self.Conv2DBNSLU(x, filters= 4, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_02 = self.Conv2DBNSLU(x_y_02, filters= 4, kernel_size=(3, 3), strides=1, activation=None, padding='same')
        
        x_y = layers.add([x_y,x_y_01,x_y_02])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        x_y = layers.UpSampling2D((2,2))(x_y)
        x_y = layers.Conv2D(filters=3, kernel_size=1, strides=(1,1), padding='same')(x_y)
        x_y = layers.Lambda(lambda x: (x - K.min(x))/(K.max(x)-K.min(x)))(x_y)
        x_y_C = utils.RotationThetaWeightLayer()([x,x_y])
        x_y = layers.concatenate([x_y_C,x])
        x_y = self.Conv2DBNSLU(x_y, filters= 8, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        #Feature extraction section
        #1 BLOCK
        x_x_33 = self.Conv2DBNSLU(x_y, filters= 8, kernel_size=(5, 5), strides=1, padding='same')
        x_x_13 = self.Conv2DBNSLU(x_y, filters= 8, kernel_size=(1, 5), strides=1, padding='same')
        x_x_31 = self.Conv2DBNSLU(x_y, filters= 8, kernel_size=(5, 1), strides=1, padding='same')
        x_y = layers.add([x_x_31,x_x_33,x_x_13])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #2 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        
        #3 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #3 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #4 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #5 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #6 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)

        #6 BLOCK
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 1024, kernel_size=(3, 3), strides=1, padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 1024, kernel_size=(1, 3), strides=1, padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 1024, kernel_size=(3, 1), strides=1, padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = layers.Activation('relu')(x_y)

        y = layers.GlobalMaxPool2D()(x_y)
        

        y_01 = layers.Dense(1024, activation='relu')(y)
        #y_01 = layers.Dense(128, activation='selu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation, name='class')(y_01)
        #convnet_model = models.Model(inputs=x, outputs=y)
        return y
    
    def ComptactModel(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        #512x512
        x_y = x
        x_y = self.Conv2DBNSLU(x, filters= 4, kernel_size=(3, 5), strides=2, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x_y, filters= 8, kernel_size=(5, 3), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_01 = self.Conv2DBNSLU(x, filters= 4, kernel_size=(5, 1), strides=2, activation='relu', padding='same')
        x_y_01 = self.Conv2DBNSLU(x_y_01, filters= 8, kernel_size=(1, 5), strides=1, activation='relu', padding='same')
        x_y_01 = layers.MaxPooling2D((2, 2), padding='same')(x_y_01)
        
        x_y_02 = self.Conv2DBNSLU(x, filters= 4, kernel_size=(3, 3), strides=2, activation='relu', padding='same')
        x_y_02 = self.Conv2DBNSLU(x_y_02, filters= 8, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_02 = layers.MaxPooling2D((2, 2), padding='same')(x_y_02)
        
        x_y = layers.concatenate([x_y,x_y_01,x_y_02])

        x_x_33 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_x_13 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(1, 5), strides=1, activation='relu', padding='same')
        x_x_31 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(5, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_x_31,x_x_33,x_x_13])
        x_y = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 96, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 96, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 96, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 96, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        y = layers.GlobalMaxPool2D()(x_y)
        y_01 = layers.Dense(512, activation='relu')(y)
        #y_01 = layers.Dense(128, activation='selu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation, name='class', kernel_regularizer=None)(y_01)
        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    
    def ComptactModel_Regional(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        #512x512
        for filter_n in [4,6,8]:
            x_y = self.Conv2DBNSLU(x, filters= filter_n, kernel_size=(3, 5), strides=2, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= filter_n, kernel_size=(5, 3), strides=1, activation='relu', padding='same')
            #x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        
        for filter_n in [4, 6,8]:
            x_y_01 = self.Conv2DBNSLU(x, filters= filter_n, kernel_size=(5, 3), strides=2, activation='relu', padding='same')
            x_y_01 = self.Conv2DBNSLU(x_y_01, filters= filter_n, kernel_size=(3, 5), strides=1, activation='relu', padding='same')
            #x_y_01 = layers.MaxPooling2D((2, 2), padding='same')(x_y_01)
        
        for filter_n in [4, 6,8]:
            x_y_02 = self.Conv2DBNSLU(x, filters= filter_n, kernel_size=(3, 3), strides=2, activation='relu', padding='same')
            x_y_02 = self.Conv2DBNSLU(x_y_02, filters= filter_n, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
           
        
        x_y = layers.add([x_y,x_y_01,x_y_02])
        x_y_02 = layers.MaxPooling2D((2, 2), padding='same')(x_y_02)
        x_x_33 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_x_13 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(1, 5), strides=1, activation='relu', padding='same')
        x_x_31 = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(5, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_x_31,x_x_33,x_x_13])
        x_y = self.Conv2DBNSLU(x_y, filters= 16, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 32, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)
        
        x_y_03 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 64, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 128, kernel_size=(1, 1), strides=1, activation='relu', padding='same')

        x_y_03 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        x_y = self.Conv2DBNSLU(x_y, filters= 256, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        #x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        #x_y = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        '''
        x_y_04 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(1, 3), strides=1, activation='relu', padding='same')
        x_y_05 = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(3, 1), strides=1, activation='relu', padding='same')
        x_y = layers.add([x_y_03,x_y_04,x_y_05])
        '''
        x_y = self.Conv2DBNSLU(x_y, filters= 1, kernel_size=(8, 8), strides=4, activation=self.final_activation, padding='same')
        y = layers.Flatten()(x_y)
        y = layers.Reshape((4,4), name='class')(y)
        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    """
    def ComptactModel_V5(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        #512x512
        x_y = x
        x_y = self.Conv2DBNSLU(x, filters= 6, kernel_size=(3, 5), strides=1, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x_y, filters= 6, kernel_size=(5, 3), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y_01 = self.Conv2DBNSLU(x, filters= 6, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_y_01 = self.Conv2DBNSLU(x_y_01, filters= 6, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_y_01 = layers.MaxPooling2D((2, 2), padding='same')(x_y_01)
        
        x_y_02 = self.Conv2DBNSLU(x, filters= 6, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_02 = self.Conv2DBNSLU(x_y_02, filters= 6, kernel_size=(3, 3), strides=1, activation='relu', padding='same')
        x_y_02 = layers.MaxPooling2D((2, 2), padding='same')(x_y_02)
        #18
        x_y = layers.concatenate([x_y,x_y_01,x_y_02])

        #36
        x_y = self.Conv2DBNSLU(x_y, filters= 36, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
        x_y = self.Conv2DBNSLU(x_y, filters= 36, kernel_size=(3, 5), strides=1, activation='relu', padding='same')
        x_y = layers.MaxPooling2D((2, 2), padding='same')(x_y)

        x_y = self.Conv2DBNSLU(x_y, filters= 512, kernel_size=(1, 1), strides=1, activation='relu', padding='same')
        y = layers.GlobalMaxPool2D()(x_y)
        y_01 = layers.Dense(512, activation='relu')(y)
        #y_01 = layers.Dense(128, activation='selu')(y)
        y = layers.Dense(self.n_class, activation=self.final_activation, name='class')(y_01)
        convnet_model = models.Model(inputs=x, outputs=y)
        return convnet_model
    """
    def ModelType_Gls4_3_Type_10(self, x):
        #shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        #x = layers.Input(shape=shape_default)
        x_y = x
        #512x512
        for filter_n in [16,32, 64]:
            x_y = self.Conv2DBNSLU(x_y, filters= filter_n, kernel_size=(5, 4), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= filter_n, kernel_size=(4, 5), strides=1, activation='relu', padding='same')
            x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
        counter=1
        for j in [128,256,512]:
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(4, 5), strides=2, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(1, 5), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(5, 1), strides=1, activation='relu', padding='same')
            x_y = self.Conv2DBNSLU(x_y, filters= j, kernel_size=(5, 5), strides=1, activation='relu', padding='same')
            if counter<5:
                x_y = layers.MaxPooling2D((2, 2), padding='valid')(x_y)
            counter +=1

        y = layers.GlobalMaxPool2D()(x_y)
        y = layers.Dense(512, activation='selu')(y)
        #y = layers.Dropout(0.2)(y)
        if self.final_activation is None:
            y = layers.Dense(self.n_class)(y)
        else: 
            y = layers.Dense(self.n_class, activation=self.final_activation, name='class')(y)
        #convnet_model = models.Model(inputs=x, outputs=y)
        return y #convnet_model
    
    def Autoencodfer(self,x):
        x_a = layers.Conv2D(filters=8, kernel_size=(3,3), strides = (1,1), padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = LeakyReLU()(x_a)
        x_a = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x_a)
        #36x4
        x_a = layers.Conv2D(filters=4, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = LeakyReLU()(x_a)
        x_a = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x_a)
        #18x4
        x_a = layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = LeakyReLU()(x_a)
        x_a_Con = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x_a)
        #36x4
        x_a = layers.UpSampling2D(size=(2,2))(x_a_Con)
        x_a = layers.Conv2D(filters=4, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a) #, kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = LeakyReLU()(x_a)
        #72x8
        x_a = layers.UpSampling2D(size=(2,2))(x_a)
        x_a = layers.Conv2D(filters=6, kernel_size=(3,3), strides=(1, 1), padding='same', kernel_constraint=min_max_norm(min_value=0.0, max_value=1))(x_a) #
        x_a = LeakyReLU()(x_a)
        #144x16
        x_a = layers.UpSampling2D(size=(2,2))(x_a)
        x_a = layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1, 1), padding='same')(x_a) #,kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        x_a = LeakyReLU()(x_a)
        x_a = layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1, 1), padding='same')(x_a) #kernel_constraint=min_max_norm(min_value=0.0, max_value=1)
        return layers.Activation('sigmoid',  name='autoencoder')(x_a), x_a_Con
    def DataInspector(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = 2)#self.initial_filter) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def DataInspector_try_2(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = self.initial_filter, length=3) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def DataInspector_try_3(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = self.initial_filter, length=2) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def DataInspector_try_4(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = self.initial_filter, depth=5) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def DataInspector_try_5(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = self.initial_filter, number_of_junctions=3) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    def DataInspector_try_6(self):
        shape_default  = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        x = layers.Input(shape=shape_default)
        x_y_o = layers.Lambda(lambda x: x/255.0)(x)
        x_y_o = utils.RotationThetaWeightLayer()([x_y_o,x_y_o])#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        x_y_o = layers.Lambda(lambda x: (2* (x - K.min(x)/(K.max(x)-K.min(x)))-1))(x_y_o)
        
        #x_a = layers.Conv2D(3, kernel_size=(1,1), kernel_constraint=min_max_norm(min_value=0.0001, max_value=0.9999999), padding='same')(x_y_o)
        #x_y_o = utils.RotationOneInputThetaWeightLayer()(x)#layers.Lambda(lambda x: K.cos(10.0) * (-2) * K.exp(-(x**2+x**2)) + K.sin(10.0) * (-2) * x * K.exp(-(x**2+x**2)))(x_y)
        #x_y_o = layers.Lambda(lambda x: (x-K.min(x)/(K.max(x)-K.min(x))))(x_y_o)
        #x_a = layers.Lambda(lambda x:(x+1)/2)(x)
        #x_y = utils.RotationThetaWeightLayer()([x_a,x_a])
        #x_y = layers.multiply([x_y,x])
        
        x_y = self.Conv2DBNSLU(x_y_o, filters= 32, kernel_size=(5, 5), strides=2, activation='relu', padding='same')
        
        #last_layer, representative_data = self.Autoencoder(x)
        #layer_x = layers.UpSampling2D((8,8))(representative_data)
        #input_x = layers.Lambda(lambda x: x[0]-K.mean(x[1]/K.mean(x[1])))([x,layer_x])
        y = self.Spider_Net(x_y, initial_filter = self.initial_filter, number_of_junctions=0) #HarmonicSeries_V4(x)
        convnet_model = models.Model(inputs=x, outputs=y)#, last_layer])
        return convnet_model
    
    def MobileNet(self):
        from keras.applications import MobileNetV2
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        base_model=MobileNetV2(weights=None,include_top=False, input_shape=shape_default, pooling='max', classes=None)
        x=base_model.output
        x=layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=layers.Dense(512,activation='relu')(x) #dense layer 2
        preds=layers.Dense(self.n_class,activation=self.final_activation)(x) #final layer with softmax activation
        model=models.Model(inputs=base_model.input,outputs=preds)
        '''
        for layer in model.layers[:10]:
            layer.trainable=False
        for layer in model.layers[10:]:
            layer.trainable=True
        '''
        return model
    def InceptionV3(self):
        from keras.applications import InceptionV3
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        base_model=InceptionV3(weights='imagenet',include_top=False, input_shape=shape_default, pooling='max', classes=None) #imports the mobilenet model and discards the last 1000 neuron layer.
        x=base_model.output
        x=layers.Dense(2048,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        #x=layers.Dense(512,activation='relu')(x) #dense layer 3
        preds=layers.Dense(self.n_class,activation=self.final_activation)(x) #final layer with softmax activation
        model=models.Model(inputs=base_model.input,outputs=preds)
        
        for layer in model.layers[:249]:
            layer.trainable = True
        for layer in model.layers[249:]:
            layer.trainable = False
        
        return model

    def DenseNet121(self):
        from keras import applications
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        base_model=applications.DenseNet121(weights='imagenet',include_top=False, input_shape=shape_default, pooling='max', classes=None) #imports the mobilenet model and discards the last 1000 neuron layer.
        x=base_model.output
        x=layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=layers.Dense(512,activation='relu')(x) #dense layer 3
        preds=layers.Dense(self.n_class,activation=self.final_activation)(x) #final layer with softmax activation
        model=models.Model(inputs=base_model.input,outputs=preds)
        for layer in model.layers[:10]:
            layer.trainable=False
        for layer in model.layers[10:]:
            layer.trainable=True
        return model
    
    def NASNetMobile(self):
        from keras import applications
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_input_channel)
        base_model=applications.NASNetMobile(weights=None,include_top=False, input_shape=shape_default, pooling='max', classes=None) #imports the mobilenet model and discards the last 1000 neuron layer.
        x=base_model.output
        x=layers.Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=layers.Dense(512,activation='relu')(x) #dense layer 2
        preds=layers.Dense(self.n_class,activation=self.final_activation)(x) #final layer with softmax activation
        model=models.Model(inputs=base_model.input,outputs=preds)
        for layer in model.layers[:5]:
            layer.trainable=False
        for layer in model.layers[5:]:
            layer.trainable=True
        return model
