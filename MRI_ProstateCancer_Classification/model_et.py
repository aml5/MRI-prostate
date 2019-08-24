#%%
import keras
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.regularizers import l2
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)

filter_size=16
input_shape=(16,144,144,1)
input_x = layers.Input(shape=input_shape,name='Input_layer', dtype = 'float32')
#1 level
x = layers.Conv3D(filters=filter_size*1, kernel_size=(1,3,3), strides = (1,1,1), padding='same')(input_x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)
x = layers.Conv3D(filters=filter_size*2, kernel_size=(1,3,3), strides=(1,1, 1), 
                                        padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.MaxPooling3D(pool_size=(1,2,2), padding='same')(x)
#2 level
x = layers.Conv3D(filters=(filter_size*4), kernel_size=(1,3,3), strides =(1,1,1), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv3D(filters=(filter_size*4), kernel_size=(1,3,3), strides =(1,1,1), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv3D(filters=(filter_size*4), kernel_size=(1,3,3), strides =(1,1,1), padding='same')(x)
#3. level
x = layers.LeakyReLU()(x)
x = layers.Conv1D(filters=filter_size*4, kernel_size=(1,1), strides=(1,1))(x)
x = layers.Reshape(target_shape=[16,-1, filter_size*4])(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(filters=filter_size*8, kernel_size=(1,1), strides=(1,1))(x)
x = layers.Reshape(target_shape=[-1, filter_size*8])(x)
x = layers.Conv1D(filters=filter_size*8, kernel_size=1, strides=1)(x)
x = layers.LeakyReLU()(x)
x = layers.Reshape(target_shape=[filter_size*32,-1])(x)
x = layers.Lambda(squash)(x)
x = layers.Reshape(target_shape=[-1,filter_size*32])(x)
x = layers.LeakyReLU()(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(2048)(x)
x = layers.LeakyReLU()(x)
#Classification
y= layers.Dense(2, activation='sigmoid', kernel_regularizer=l2(0.01))(x)
model = Model(inputs=input_x, outputs=y)
model.summary()