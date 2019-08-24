import keras
from keras import layers, models
from keras.layers.merge import concatenate, add, subtract
import keras.backend as K
from keras.regularizers import L1L2
import numpy as np
from keras import optimizers, callbacks 
from keras.layers import Lambda, Input, Dense
from keras.losses import mse
from keras.utils import plot_model

class ModelAbstract():
    def __init__(self, input_shape=(256,256), 
                        number_of_class=3,number_of_channel=1, 
                        depth_for_encoder=4, ratio_for_decline=0.5, 
                        n_filter=32, activation_last='sigmoid',
                        save_dir='./', batch_size=16, lr=0.1, 
                        debug=False,epoch=50,lr_factor=2, min_lr=1e-5,
                        patience=5, monitor='val_loss', mode='max',
                        model=None, metrics=['mse'], loss='loss',optimizer=keras.optimizers.Adam):
        self.input_shape = input_shape
        self.number_of_class = number_of_class
        self.number_of_channel= number_of_channel
        self.depth_for_encoder = depth_for_encoder
        self.ratio_for_decline = ratio_for_decline
        self.n_filter = n_filter
        self.activation_last = activation_last
        self.model = None
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.debug = debug
        self.min_lr = min_lr
        self.lr_factor= lr_factor
        self.patience = patience
        self.monitor= monitor
        self.mode= mode
        self.model = model
        self.loss= loss
        self.metrics = metrics
        self.optimizer =optimizer
    def Load(self):
        self.model.compile(optimizer=self.optimizer(lr=self.lr),loss=self.loss, metrics=self.metrics)
    def GenerateModel(self):
        pass
    '''
    def Train(self, keras_modus=True, optimizer=optimizers.Adagrad,loss=['mse'], metrics=["acc", "mse"], training_set_pop=20000, valid_set_pop=5000, train_input_generator=None,valid_input_generator=None):
        if keras_modus:
            log = callbacks.CSVLogger(self.save_dir + '/log.csv')
            tb = callbacks.TensorBoard(log_dir=self.save_dir + '/tensorboard-logs',
                                    batch_size=self.batch_size, histogram_freq=self.debug)
    
            lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: self.lr * (0.9 **self.epoch))
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=self.monitor, mode=self.mode, factor=self.lr_factor, patience=self.patience, min_lr=self.min_lr,verbose=1)
            checkpoint = callbacks.ModelCheckpoint(self.save_dir + '/weights-{epoch:02d}.h5', monitor=self.monitor,
                                            save_best_only=True, save_weights_only=True, mode=self.mode,verbose=1)
            
            train_steps_per_epoch = training_set_pop // self.batch_size #train_img_dataset.shape[0]
            valid_steps_per_epoch = valid_set_pop // self.batch_size
            print("Proc: training...")
            self.model.fit_generator(generator=train_input_generator,
                                    steps_per_epoch=train_steps_per_epoch,
                                    epochs=self.epoch,
                                    use_multiprocessing=True,
                                    validation_steps=valid_steps_per_epoch,
                                    validation_data=valid_input_generator,
                                    callbacks=[log, tb, checkpoint,lr_decay, reduce_lr])
            print("Done: training...")
        else:
            pass
    
    def Test(self):
        pass
    '''
    def Conv2DBNSLU(self, x, filters, kernel_size=1, strides=1, padding='same', activation="relu", name="", bias=False, bn=True, scale=False):
        x = layers.Conv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=bias)(x)
        if bn:
            x = layers.BatchNormalization(scale=scale)(x)
        x = layers.Activation(activation)(x)
        return x
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self, z_mean,z_log_var):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE_Model(ModelAbstract):
    def Load(self, intermediate_dim = 512, batch_size = 128, latent_dim = 2):
        # VAE model = encoder + decoder
        # build encoder model
        original_dim = self.input_shape[0] * self.input_shape[1]
        inputs = Input(shape=self.input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = models.Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        plot_model(self.decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = models.Model(inputs, outputs, name='vae_mlp')
        self.models = (self.encoder, self.decoder)
    '''
    def Train(self):
        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = mse(self.inputs, self.outputs)
        else:
            reconstruction_loss = binary_crossentropy(self.inputs,
                                                    self.outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
        plot_model(vae,
                to_file='vae_mlp.png',
                show_shapes=True)

        if args.weights:
            vae.load_weights(args.weights)
        else:
            # train the autoencoder
            vae.fit(x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, None))
            vae.save_weights('vae_mlp_mnist.h5')
    def Test(self):
        pass
    '''

class AutoEncoderModel(ModelAbstract):
    def Block(self, x, filter_number,factor_kernel, pool=False):
        conv_level = layers.Conv2D(filters=filter_number, kernel_size=factor_kernel, strides=2,
                                padding='same', activation='selu', kernel_regularizer=L1L2(0.01,0.01))(x)
        if pool:
            conv_level = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level)
        return conv_level

    def DeConv(self, x, filter_number,factor_kernel):
        conv_level = keras.layers.Conv2DTranspose(filters=filter_number, kernel_size=factor_kernel, strides=2, padding="same")(x)
        return conv_level
    
    def Load(self):
        x = layers.Input(shape=self.input_shape)
        #Encode
        n_filter_c = self.n_filter
        y = x
        for i in range(self.depth_for_encoder):
            if i==0:
                y =  self.Block(y, n_filter_c,3)
            else:
                n_filter_c= int(round(n_filter_c*self.ratio_for_decline))
                y =  self.Block(y, n_filter_c,3)
        
        #Decode
        for i in range(self.depth_for_encoder):
            if i==0:
                y = self.DeConv(y, n_filter_c, 3)
            else:
                n_filter_c= int(round(n_filter_c/self.ratio_for_decline))
                y =  self.DeConv(y, n_filter_c,3)
        y = layers.Conv2D(filters=self.n_filter, kernel_size=1, strides=1,
                                        padding='same', activation=self.activation_last)(x)
        train_model = models.Model(x, y)
        return train_model
    def Train(self):
        pass
    def Test(self):
        pass

class ConvModel_V2(ModelAbstract):
    def __init__(self):
        super().__init__()
    
    ######
    #
    #                    input_shape=(256,256), 
    #                    number_of_class=3,number_of_channel=1, 
    #                    depth_for_encoder=4, ratio_for_decline=0.5, 
    #                    n_filter=32, activation_last='sigmoid',
    #                    save_dir='./', batch_size=16, lr=0.1, 
    #                    debug=False,epoch=50,lr_factor=2, min_lr=1e-5,
    #                    patience=5, monitor='val_loss', mode='max',
    #                    model=None, metrics=['mse'], loss='loss',optimizer=keras.optimizers.Adam    #
    #######
    def GenerateModel(self):
        ########
        #
        #
        #######
        init_X = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
        filter_size =self.n_filter*5
        model = models.Sequential()
        shape_default = (None,self.input_shape[0], self.input_shape[1],self.number_of_channel)
        model.add(layers.SeparableConv2D(filters=filter_size, input_shape=shape_default, kernel_size=(2,2), strides=(1, 1), 
                                padding='same', activation='selu', use_bias=True, depthwise_initializer=init_X, pointwise_initializer=init_X))
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
        for i in [3,4,5]:
            model.add(layers.Conv2D(filters=(filter_size//i), 
                                    kernel_size = (3,3),
                                    strides=1,
                                    padding='same'))
            model.add(layers.BatchNormalization(scale=True))
            model.add(layers.Activation('relu'))
            model.add(layers.MaxPooling2D((2, 2), padding='valid'))
        
        model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(self.number_of_class, activation=self.activation_last))
        model.compile(optimizer=self.optimizer(lr=self.lr),loss=self.loss, metrics=self.metrics)
        model.summary()
        return model

