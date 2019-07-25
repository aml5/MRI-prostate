#%%
from keras import layers
from keras.layers import Conv3D, Conv3DTranspose, Input, MaxPooling3D, Concatenate, Activation
from keras import optimizers
from advanced_activations import SineReLU
from keras import regularizers
from keras.models import Sequential, Model
import gc
import yogi
class Model_Storage:
    #['mse', 'acc', dice_coef, recall_at_thresholds, precision_at_thresholds, auc_roc]
    #optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0.000001,lr=lr)
    #keras.optimizers.SGD(lr=lr, momentum=0.90, decay=decay, nesterov=False)
    #opt_noise = add_gradient_noise(optimizers.Adam)
    #optimizer = opt_noise(lr, amsgrad=True)#, nesterov=True)#opt_noise(lr, amsgrad=True)
    def __init__(self, 
                n_filter=16, 
                number_of_class=1, 
                input_shape=(20,384,384,1), 
                activation_last='softmax', 
                metrics=['acc', 'mse'], 
                loss='binary_crossentropy', 
                dropout=0.05, 
                init='glorot_uniform', 
                optimizer='adam',
                two_output=False):
        self.n_filter = n_filter
        self.number_of_class = number_of_class
        self.input_shape = input_shape
        self.activation_last = activation_last
        self.metrics = metrics
        self.dropout = dropout
        self.init = init
        self.loss = loss
        self.two_output = two_output
        self.optimizer = optimizer

    def Compile(self, model):
        model.compile(optimizer=self.optimizer,loss=self.loss, metrics=self.metrics)
        return model     

    def D3_Model_Detection(self):
        filter_size =self.n_filter
        input_x = layers.Input(shape=self.input_shape,name='Input_layer', dtype = 'float32')
        #1 level
        x = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides = (1,1,1), kernel_initializer=self.init, padding='same')(input_x)
        #x = SineReLU()(x)
        x = layers.Conv3D(filters=filter_size*2, kernel_size=(3,3,3), strides=(1,1, 1), 
                                            padding='same',kernel_initializer=self.init)(x)
        #x = SineReLU()(x)
        x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
        #2 level
        conv_list = []
        counter = 0
        for index, kernel_sizes in enumerate([
                                    [(1,3,3), (1,1,3)], 
                                    [(3,3,3), (3,1,3)],
                                    [(3,3,1), (1,3,1)]
                                    ]):
            for kernel_size in (kernel_sizes):
                x = layers.Conv3D(filters=(filter_size*4), kernel_size=kernel_size, kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
                x = layers.BatchNormalization()(x)
                #x = SineReLU()(x)
                #x = layers.SpatialDropout3D(dropout)(x)
                counter = counter+1
            conv_list.append(x)
        x = layers.add(conv_list)
        x = layers.Conv3D(filters=filter_size*8, kernel_size=(3,3,3), strides=(2,2, 2), kernel_initializer=self.init, 
                                            padding='same')(x)
        x = layers.Reshape(target_shape=[4,-1, filter_size*8])(x)
        x = layers.Conv2D(filters=filter_size*8, kernel_size=(1,1296), kernel_initializer=self.init, strides=(1,1296))(x)
        x = layers.BatchNormalization()(x)
        #x = SineReLU()(x)
        x = layers.Reshape(target_shape=[filter_size*8,-1])(x)
        x = layers.Conv1D(filters=self.number_of_class, kernel_size=filter_size*8, strides=filter_size*8, kernel_initializer=self.init, activation=self.activation_last)(x)
        y = layers.Flatten()(x)
        #Classification    
        model = Model(inputs=input_x, outputs=y)
        return model

    def D3_Model_Segmentation_old(self):
        filter_size =self.n_filter
        input_x = layers.Input(shape=self.input_shape,name='Input_layer', dtype = 'float32')
        #1 level
        x_01 = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides = (1,1,1), kernel_initializer=self.init, padding='same')(input_x)
        x = SineReLU()(x_01)
        x = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides=(2,2, 2), 
                                            padding='same',kernel_initializer=self.init)(x_01)
        x = SineReLU()(x)
        #x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
        #2 level
        conv_list = []
        counter = 0
        #Down Sampling
        kernel_size_d = [[(1,3,3), (1,1,3)], 
                        [(3,3,3), (3,1,3)],
                        [(3,3,1), (1,3,1)],
                        [(3,3,3),(3,3,3),(3,3,3)],
                        [(1,3,3), (3,1,3), (3,3,1), (3,3,3)]
                        #[(1,2,2), (2,1,2), (2,2,1), (2,2,2)]
                        ]
        
        for index, kernel_sizes in enumerate(kernel_size_d):
            for kernel_size in (kernel_sizes):
                x = layers.Conv3D(filters=(filter_size*(1+index)), kernel_size=kernel_size, kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
                x = layers.BatchNormalization(scale=False)(x)
                x = SineReLU()(x)
                x = layers.SpatialDropout3D(self.dropout)(x)

                counter = counter+1
            if index==0 and index<6:
                x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
            elif index < (len(kernel_size_d)-1):
                x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
            conv_list.append(x)
        x = conv_list[-1]
        #Up Sampling
        counter = 0
        kernel_sizes_c = [#[(1,2,2)],
                          [(1,3,3)],
                          [(3,3,3)],
                          [(3,3,1)],
                          [(3,3,3)],
                          [(1,3,3)]]
        reverse_index = len(kernel_sizes_c) -1
        for index, kernel_sizes in enumerate(kernel_sizes_c):
            for kernel_size in (kernel_sizes):
                stride_vle = (1,2,2)
                if index==(len(kernel_sizes_c)-1):
                    stride_vle = (2,2,2)
                if index>0 and reverse_index>=1:
                    reverse_index_c = reverse_index -1
                    x = layers.concatenate([x, conv_list[reverse_index_c]]) #concatenate
                
                x = layers.Conv3DTranspose(filters=(filter_size*(1+reverse_index)), kernel_size=kernel_size, kernel_initializer=self.init, strides=stride_vle, padding='same', name='UpSampling_%s' % (counter))(x)
                
                x = SineReLU()(x)
                x = layers.SpatialDropout3D(self.dropout)(x)
                
                reverse_index = reverse_index -1
                counter = counter+1
        #size=(2,1,1))(x)#
        x = layers.Conv3DTranspose(filters=(filter_size*2), kernel_size=(2,1,1), kernel_initializer=self.init, strides=(2,1,1), padding='same')(x)
        x = SineReLU()(x)
        #x = layers.concatenate([x_01, x])
        y = layers.Conv3D(filters=self.number_of_class, 
                                    kernel_size=(3,3,3),
                                    activation=self.activation_last, 
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    padding='same', 
                                    name='output')(x)
        '''
        x_c = layers.concatenate([x,input_x])
        x = layers.Conv3D(filters=self.n_filter, 
                                    kernel_size=(3,3,3),
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    padding='same', 
                                    name='input_2')(x_c)
        x = SineReLU()(x)
        #256, 128, 64, 32
        #Classification    
        for i in range(4):
            index = i + 1
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_2_0_%s' % (index))(x)
            #x = layers.BatchNormalization(scale=False)(x)
            x = SineReLU()(x)
            
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_2_1_%s' % (index))(x)
            #x = layers.BatchNormalization(scale=False)(x)
            x = SineReLU()(x)
            
            if i >=2:
                x = layers.SpatialDropout3D(self.dropout)(x)
            x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

        for i in range(4):
            index = 4-i
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides=(1,1,1), padding='same', name='Conv3D_2_T_%s' % (index))(x)
            x = SineReLU()(x)

            x = layers.Conv3DTranspose(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides=(2,2,2), padding='same', name='UpSampling_2_%s' % (index))(x)
            x = SineReLU()(x)
        y = layers.Conv3D(filters=self.number_of_class, 
                                    kernel_size=(3,3,3),
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    activation=self.activation_last,
                                    padding='same', 
                                    name='final_output')(x)
        '''
        model = Model(inputs=input_x, outputs=y)
        
        return model

    def D3_Model_Segmentation(self):
        filter_size =self.n_filter
        input_x = layers.Input(shape=self.input_shape,name='Input_layer', dtype = 'float32')
        #1 level
        x_01 = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides = (1,1,1), kernel_initializer=self.init, padding='same')(input_x)
        x = layers.LeakyReLU()(x_01)
        x = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides=(2,2, 2), 
                                            padding='same',kernel_initializer=self.init)(x_01)
        x = layers.LeakyReLU()(x)
        #x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
        #2 level
        conv_list = []
        counter = 0
        #Down Sampling
        '''
        kernel_size_d = [[(1,3,3), (1,1,3)], 
                        [(3,3,3), (3,1,3)],
                        [(3,3,1), (1,3,1)],
                        [(3,3,3),(3,3,3),(3,3,3)],
                        [(1,3,3), (3,1,3), (3,3,1), (3,3,3)]
                        #[(1,2,2), (2,1,2), (2,2,1), (2,2,2)]
                        ]

        '''
        kernel_size_d = [[(3,3,3), (3,3,3)], 
                        [(3,3,3), (3,3,3)],
                        [(3,3,3), (3,3,3)],
                        [(3,3,3),(3,3,3),(3,3,3)],
                        [(3,3,3), (3,3,3),(3,3,3)]
                        #[(1,2,2), (2,1,2), (2,2,1), (2,2,2)]
                        ]
        
        for index, kernel_sizes in enumerate(kernel_size_d):
            for kernel_size in (kernel_sizes):
                x = layers.Conv3D(filters=(filter_size*(1+index)), kernel_size=kernel_size, kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
                x = layers.BatchNormalization(scale=False)(x)
                x = layers.LeakyReLU()(x)
                #x = layers.SpatialDropout3D(self.dropout)(x)

                counter = counter+1
            if index==0 and index<6:
                x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
            elif index < (len(kernel_size_d)-1):
                x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
            conv_list.append(x)
        x = conv_list[-1]
        #Up Sampling
        counter = 0
        '''
        kernel_sizes_c = [#[(1,2,2)],
                          [(1,3,3)],
                          [(3,3,3)],
                          [(3,3,1)],
                          [(3,3,3)],
                          [(1,3,3)]]
        '''
        kernel_sizes_c = [#[(1,2,2)],
                          [(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)]]
        reverse_index = len(kernel_sizes_c) -1
        for index, kernel_sizes in enumerate(kernel_sizes_c):
            for kernel_size in (kernel_sizes):
                stride_vle = (1,2,2)
                if index==(len(kernel_sizes_c)-1):
                    stride_vle = (2,2,2)
                if index>0 and reverse_index>=1:
                    reverse_index_c = reverse_index -1
                    x = layers.concatenate([x, conv_list[reverse_index_c]]) #concatenate
                
                x = layers.Conv3DTranspose(filters=(filter_size*(1+reverse_index)), kernel_size=kernel_size, kernel_initializer=self.init, strides=stride_vle, padding='same', name='UpSampling_%s' % (counter))(x)
                
                x = layers.LeakyReLU()(x)
                #x = layers.SpatialDropout3D(self.dropout)(x)
                
                reverse_index = reverse_index -1
                counter = counter+1
        #size=(2,1,1))(x)#
        #x = layers.Conv3DTranspose(filters=(filter_size), kernel_size=(2,1,1), kernel_initializer=self.init, strides=(2,1,1), padding='same')(x)
        x = layers.UpSampling3D(size=(2,1,1))(x)
        #x = layers.LeakyReLU()(x)
        #
        y = layers.Conv3D(filters=self.number_of_class, 
                                    kernel_size=(3,3,3),
                                    activation=self.activation_last, 
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    padding='same', 
                                    name='output')(x)
        '''
        x_c = layers.concatenate([x,input_x])
        x = layers.Conv3D(filters=self.n_filter, 
                                    kernel_size=(3,3,3),
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    padding='same', 
                                    name='input_2')(x_c)
        x = SineReLU()(x)
        #256, 128, 64, 32
        #Classification    
        for i in range(4):
            index = i + 1
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_2_0_%s' % (index))(x)
            #x = layers.BatchNormalization(scale=False)(x)
            x = SineReLU()(x)
            
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_2_1_%s' % (index))(x)
            #x = layers.BatchNormalization(scale=False)(x)
            x = SineReLU()(x)
            
            if i >=2:
                x = layers.SpatialDropout3D(self.dropout)(x)
            x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

        for i in range(4):
            index = 4-i
            x = layers.Conv3D(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides=(1,1,1), padding='same', name='Conv3D_2_T_%s' % (index))(x)
            x = SineReLU()(x)

            x = layers.Conv3DTranspose(filters=self.n_filter*index, kernel_size=(3,3,3), kernel_initializer=self.init, strides=(2,2,2), padding='same', name='UpSampling_2_%s' % (index))(x)
            x = SineReLU()(x)
        y = layers.Conv3D(filters=self.number_of_class, 
                                    kernel_size=(3,3,3),
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    activation=self.activation_last,
                                    padding='same', 
                                    name='final_output')(x)
        '''
        model = Model(inputs=input_x, outputs=y)
        
        return model
    
    def side_branch(self, x, factor):
        x = Conv3D(1, (1,1, 1), activation=None, padding='same')(x)

        kernel_size = (1, 2*factor, 2*factor)
        x = Conv3DTranspose(1, kernel_size, strides=(1,factor,factor), padding='same', use_bias=False, activation=None)(x)
        return x

    def D3_EdgeProstateSegmentation(self):
        img_input = layers.Input(shape=self.input_shape,name='Input_layer', dtype = 'float32')

        # Block 1
        x = Conv3D(16, (3,3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv3D(16, (3,3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        b1= self.side_branch(x, 1) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block1_pool')(x) # 240 240 64

        # Block 2
        x = Conv3D(32, (2,3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv3D(32, (2,3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b2= self.side_branch(x, 2) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block2_pool')(x) # 120 120 128

        # Block 3
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        b3= self.side_branch(x, 4) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block3_pool')(x) # 60 60 256

        # Block 4
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        b4= self.side_branch(x, 8) # 480 480 1
        x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding='same', name='block4_pool')(x) # 30 30 512

        # Block 5
        x = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
        b5= self.side_branch(x, 16) # 480 480 1

        # fuse
        fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
        fuse = Conv3D(1, (1,1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

        # outputs
        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        o5 = Activation('sigmoid', name='o5')(b5)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        # model
        model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
        return model

    def ProstateD3Segmentation(self, input_x):
        filter_size =self.n_filter
        
        #1 level
        x_01 = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides = (1,1,1), kernel_initializer=self.init, padding='same')(input_x)
        x = layers.LeakyReLU()(x_01)
        x = layers.Conv3D(filters=filter_size, kernel_size=(3,3,3), strides=(2,2, 2), 
                                            padding='same',kernel_initializer=self.init)(x_01)
        x = layers.LeakyReLU()(x)
        #x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
        #2 level
        conv_list = []
        counter = 0
        #Down Sampling
        kernel_size_d = [[(3,3,3), (3,3,3)], 
                        [(3,3,3), (3,3,3)],
                        [(3,3,3), (3,3,3)],
                        [(3,3,3),(3,3,3),(3,3,3)],
                        [(3,3,3), (3,3,3),(3,3,3)]
                        ]
        
        for index, kernel_sizes in enumerate(kernel_size_d):
            for kernel_size in (kernel_sizes):
                x = layers.Conv3D(filters=(filter_size*(1+index)), kernel_size=kernel_size, kernel_initializer=self.init, strides =(1,1,1), padding='same', name='Conv3D_%s' % (counter))(x)
                x = layers.BatchNormalization(scale=False)(x)
                x = layers.LeakyReLU()(x)
                x = layers.SpatialDropout3D(self.dropout)(x)

                counter = counter+1
            if index==0 and index<6:
                x = layers.MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
            elif index < (len(kernel_size_d)-1):
                x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
            conv_list.append(x)
        x = conv_list[-1]
        #Up Sampling
        counter = 0
        
        kernel_sizes_c = [[(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)],
                          [(3,3,3)]]

        reverse_index = len(kernel_sizes_c) -1
        for index, kernel_sizes in enumerate(kernel_sizes_c):
            for kernel_size in (kernel_sizes):
                stride_vle = (1,2,2)
                if index==(len(kernel_sizes_c)-1):
                    stride_vle = (2,2,2)
                if index>0 and reverse_index>=1:
                    reverse_index_c = reverse_index -1
                    x = layers.concatenate([x, conv_list[reverse_index_c]]) #concatenate
                
                x = layers.Conv3DTranspose(filters=(filter_size*(1+reverse_index)), kernel_size=kernel_size, kernel_initializer=self.init, strides=stride_vle, padding='same', name='UpSampling_%s' % (counter))(x)
                
                x = layers.LeakyReLU()(x)
                x = layers.SpatialDropout3D(self.dropout)(x)
                
                reverse_index = reverse_index -1
                counter = counter+1
        #size=(2,1,1))(x)#
        #x = layers.Conv3DTranspose(filters=(filter_size), kernel_size=(2,1,1), kernel_initializer=self.init, strides=(2,1,1), padding='same')(x)
        x = layers.UpSampling3D(size=(2,1,1))(x)
        #x = layers.LeakyReLU()(x)
        #
        y = layers.Conv3D(filters=self.number_of_class, 
                                    kernel_size=(3,3,3),
                                    activation=self.activation_last, 
                                    kernel_initializer=self.init, 
                                    strides=(1,1,1), 
                                    padding='same', 
                                    name='output')(x)
        #model = Model(inputs=input_x, outputs=y)
        
        return y
    
    def EdgeProstate(self, img_input):
        # Block 1
        x = Conv3D(16, (3,3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv3D(16, (3,3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        b1= self.side_branch(x, 1) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block1_pool')(x) # 240 240 64

        # Block 2
        x = Conv3D(32, (2,3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv3D(32, (2,3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b2= self.side_branch(x, 2) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block2_pool')(x) # 120 120 128

        # Block 3
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv3D(64, (1, 3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        b3= self.side_branch(x, 4) # 480 480 1
        x = MaxPooling3D((1,2, 2), strides=(1,2, 2), padding='same', name='block3_pool')(x) # 60 60 256

        # Block 4
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv3D(128, (1, 3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        b4= self.side_branch(x, 8) # 480 480 1
        x = MaxPooling3D((1, 2, 2), strides=(1, 2, 2), padding='same', name='block4_pool')(x) # 30 30 512

        # Block 5
        x = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv3D(256, (1, 3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
        b5= self.side_branch(x, 16) # 480 480 1

        # fuse
        fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
        fuse = Conv3D(1, (1,1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1

        # outputs
        o1 = Activation('sigmoid', name='o1')(b1)
        o2 = Activation('sigmoid', name='o2')(b2)
        o3 = Activation('sigmoid', name='o3')(b3)
        o4 = Activation('sigmoid', name='o4')(b4)
        o5 = Activation('sigmoid', name='o5')(b5)
        ofuse = Activation('sigmoid', name='ofuse')(fuse)

        # model
        #model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
        return o1, o2, o3, o4, o5, ofuse

    def Prostate_D3_Segmentation(self):
        input_x = layers.Input(shape=self.input_shape,name='Input_layer_0', dtype = 'float32')
        o1, o2, o3, o4, o5, ofuse = self.EdgeProstate(input_x)
        new_input =layers.Lambda(lambda x: (x[0] - x[1])**2)([ofuse, input_x])
        y_00 = self.ProstateD3Segmentation(new_input)
        return Model(inputs=[input_x], outputs=[o1, o2, o3, o4, o5, ofuse, y_00])
#%%
#mi = Model_Storage()
#d = mi.D3_Model_Segmentation()
#d.summary()