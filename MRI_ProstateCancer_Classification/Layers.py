#%%
import keras
from keras import layers, models
#%%
import numpy as np
a = np.ones((3,4, 4))

# npad is a tuple of (n_before, n_after) for each dimension
npad = ((0,0),(1, 1), (1, 1))
b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)

print(b.shape)
print(b)


from keras import backend as K
from keras.layers import Layer

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class AttentionLayer(Layer):
    def __init__(self, bias=False, **kwargs):
        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape) == 4
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1],input_shape[-1]),
                                initializer='uniform',
                                trainable=True)
        self.u = self.add_weight(name='U', 
                                      shape=(input_shape[-1],input_shape[-1]),
                                      initializer='uniform',
                                      trainable=True)

        if self.bias:
            self.b = self.add_weight(
                                    name='B',
                                    shape=(input_shape[-1],),
                                    initializer='zero'
                                    )
        
        
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        uit = K.dot(x, self.W)#dot_product(x, self.W)
        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)
        a = K.exp(ait)

        a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
        #a = K.expand_dims(a)
        weighted_input = x * a
        print('weighted_input',weighted_input)
        return weighted_input #K.sum(weighted_input, axis=1, keepdims=True)
    def compute_output_shape(self, input_shape):
        return input_shape# (input_shape[0], input_shape[-1],input_shape[-3])