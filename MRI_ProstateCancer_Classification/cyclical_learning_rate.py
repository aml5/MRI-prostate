from keras.callbacks import Callback
from keras import backend as K
import numpy as np
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import Layer
from keras.layers import InputSpec
from keras import backend as K
from keras.utils import get_custom_objects

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular2',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            Peak_Location='default'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        if Peak_Location not in ['default', 'LowFirst',
                        'HighFirst', 'PeakAtEnd']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.Peak_Location = Peak_Location
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        if self.Peak_Location=='default' or self.Peak_Location=='HighFirst':
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.Peak_Location=='LowFirst':
            x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
            x = 1 -x
        if self.Peak_Location=='PeakAtEnd':
            x = self.clr_iterations / self.step_size - 2 * cycle + 1
            
        #'PeakAtBegin', 'PeakAtEnd'
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class SineReLU(Layer):
    """Sine Rectified Linear Unit to generate oscilations.

    It allows an oscilation in the gradients when the weights are negative.
    The oscilation can be controlled with a parameter, which makes it be close
    or equal to zero. The functional is diferentiable at any point due to
    its derivative.
    For instance, at 0, the derivative of 'sin(0) - cos(0)'
    is 'cos(0) + sin(0)' which is 1.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        epsilon: float. Hyper-parameter used to control the amplitude of the
            sinusoidal wave when weights are negative.
            The default value, 0.0025, since it works better for CNN layers and
            those are the most used layers nowadays.
            When using Dense Networks, try something around 0.006.

    # References:
        - [SineReLU: An Alternative to the ReLU Activation Function](
           https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d).

        This function was
        first introduced at the Codemotion Amsterdam 2018 and then at
        the DevDays, in Vilnius, Lithuania.
        It has been extensively tested with Deep Nets, CNNs,
        LSTMs, Residual Nets and GANs, based
        on the MNIST, Kaggle Toxicity and IMDB datasets.

    # Performance:

        - Fashion MNIST
          * Mean of 6 runs per Activation Function
            * Fully Connection Network
              - SineReLU: loss mean -> 0.3522; accuracy mean -> 89.18;
                  mean of std loss -> 0.08375204467435822
              - LeakyReLU: loss mean-> 0.3553; accuracy mean -> 88.98;
              mean of std loss -> 0.0831161868455245
              - ReLU: loss mean -> 0.3519; accuracy mean -> 88.84;
              mean of std loss -> 0.08358816501301362
            * Convolutional Neural Network
              - SineReLU: loss mean -> 0.2180; accuracy mean -> 92.49;
              mean of std loss -> 0.0781155784858847
              - LeakyReLU: loss mean -> 0.2205; accuracy mean -> 92.37;
              mean of std loss -> 0.09273670474788205
              - ReLU: loss mean -> 0.2144; accuracy mean -> 92.45;
              mean of std loss -> 0.09396114585977
        - MNIST
          * Mean of 6 runs per Activation Function
            * Fully Connection Network
              - SineReLU: loss mean -> 0.0623; accuracy mean -> 98.53;
              mean of std loss -> 0.06012015231824904
              - LeakyReLU: loss mean-> 0.0623; accuracy mean -> 98.50;
              mean of std loss -> 0.06052147632835356
              - ReLU: loss mean -> 0.0605; accuracy mean -> 98.49;
              mean of std loss -> 0.059599885665016096
            * Convolutional Neural Network
              - SineReLU: loss mean -> 0.0198; accuracy mean -> 99.51;
              mean of std loss -> 0.0425338329550847
              - LeakyReLU: loss mean -> 0.0216; accuracy mean -> 99.40;
              mean of std loss -> 0.04834468835196667
              - ReLU: loss mean -> 0.0185; accuracy mean -> 99.49;
              mean of std loss -> 0.05503719489690131

    # Jupyter Notebooks
        - https://github.com/ekholabs/DLinK/blob/master/notebooks/keras

    # Examples
        The Advanced Activation function SineReLU have to be imported from the
        keras_contrib.layers package.

        To see full source-code of this architecture and other examples,
        please follow this link: https://github.com/ekholabs/DLinK

        ```python
            model = Sequential()
            model.add(Dense(128, input_shape = (784,)))
            model.add(SineReLU())
            model.add(Dropout(0.2))

            model.add(Dense(256))
            model.add(SineReLU())
            model.add(Dropout(0.3))

            model.add(Dense(1024))
            model.add(SineReLU())
            model.add(Dropout(0.5))

            model.add(Dense(10, activation = 'softmax'))
        ```
    """

    def __init__(self, epsilon=0.0025, **kwargs):
        super(SineReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = K.cast_to_floatx(epsilon)

    def call(self, Z):
        m = self.epsilon * (K.sin(Z) - K.cos(Z))
        A = K.maximum(m, Z)
        return A

    def get_config(self):
        config = {'epsilon': float(self.epsilon)}
        base_config = super(SineReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

get_custom_objects().update({'SineReLU': SineReLU})