#%
import keras
import SimpleITK
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import callbacks

##############
# 
###############
#%
def D3GenerateModel(n_filter=16, number_of_class=2, input_shape=(16,144,144,1),activation_last='sigmoid', metrics=['mse', 'acc'],loss='mse', optimizer='adam',dropout=0.5, init='glorot_uniform'):
    filter_size =n_filter
    model = Sequential()
    model.add(layers.Conv3D(filters=filter_size, input_shape=input_shape,  kernel_size=(3,3,3), strides=(1,1, 1), 
                                padding='valid', activation='selu'))
    model.add(layers.Conv3D(filters=filter_size*2, kernel_size=(3,3,3), strides=(1, 2,2), 
                                padding='valid', activation='selu'))
    model.add(layers.MaxPooling3D((1, 2,2), padding='valid'))
    model.add(layers.Conv3D(filters=filter_size*2, kernel_size=(3,3,3), strides=(1,1,1), 
                                padding='valid', activation='selu'))
    model.add(layers.Conv3D(filters=filter_size*4, kernel_size=(3,3,3), strides=(1, 2,2), 
                                padding='valid', activation='selu'))
    model.add(layers.MaxPooling3D((1, 2,2), padding='valid'))
    model.add(layers.Conv3D(filters=filter_size*4, kernel_size=(3,3,3), strides=(1,1, 1), 
                                padding='valid', activation='selu'))
    model.add(layers.Conv3D(filters=filter_size*8, kernel_size=(3,3,3), strides=(1, 2,2), 
                                padding='valid', activation='selu'))
    model.add(layers.MaxPooling3D((1,2, 2), padding='same'))
    model.add(layers.Conv3D(filters=filter_size*16, kernel_size=(3,3,3), strides=(1,1, 1), 
                                padding='same', activation='selu'))
    model.add(layers.Conv3D(filters=filter_size*32, kernel_size=(3,3,3), strides=(2,2, 2), 
                                padding='same', activation='selu'))
    
    #model.add(layers.MaxPooling2D((2, 2), padding='valid'))
    model.add(layers.GlobalMaxPooling3D())
    #Encoder
    model.add(layers.Dense(512, activation='selu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='selu'))
    model.add(layers.Dense(2, activation='softmax'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.summary()
    model.compile(optimizer=keras.optimizers.adam(lr=2e-6),loss='categorical_crossentropy', metrics=metrics)
    return model

D3GenerateModel()