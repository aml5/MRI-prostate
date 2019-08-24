"""
Author: Okyaz Eminaga
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import argparse
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
from keras.utils.vis_utils import plot_model
import numpy as np
from keras import initializers
import scipy
import gc
import keras
import os
#Get the list of negative images
import pickle
import math
from keras.utils import multi_gpu_model
from convaware import ConvolutionAware
#from keras_gradient_noise import add_gradient_noise
import cyclical_learning_rate
import matplotlib.pyplot as plt
import keras.optimizers
from keras import regularizers
from keras import backend as K
from keras.regularizers import l2
import model_storage
from capsule_layer import Capsule
import utils
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
import configuration
#######################
#   CONFIGURATION
#######################
lr=1e-4#5e-5 #5e-4
epochs = 30
decay = lr/epochs
counter_s=0
gc.collect()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.CUDA_VISIBLE_DEVICES
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#from tensorflow.python import debug as tf_debug
#config.gpu_options.allow_growth = True
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#set_session(sess)

#set_session(tf.Session(config=config))
##########################
# Run
##########################
class ModelRunner():
    def __init__(self, args):
        """
        A mini Capsule Network on histology images.
        :param input_shape: data shape, 3d, [width, height, channels], channels= R,G,B or H,E,A/B
        :param args: configuration data
        :param target_size: target_size of the image
        :param input_shape: define the input shape of the model
        :param cropping_size: apply to cropping size
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
        `eval_model` can also be used for training.
        """
        self.args = args

    def Run(self):
        loader = model_storage.Models(input_shape=configuration.input_shape, 
                                                        batch_size=configuration.batch_size,
                                                        number_input_channel=configuration.number_input_channel,
                                                        final_activation=configuration.final_activation,
                                                        n_class=configuration.n_class,
                                                        two_output=configuration.Two_output
                                                        )
        if (configuration.parallel):
            with tf.device('/cpu:0'):
                method_to_call = getattr(loader, configuration.model_name) #'CancerDetection_HarmonicSeries')
                model = method_to_call()
        else:
            method_to_call = getattr(loader, configuration.model_name) #'CancerDetection_HarmonicSeries')
            model = method_to_call()
         
        if self.args.weights is None and self.args.load_previous_weight:
            print('weight file is not given. Searching for the last weight in the given folder %s...' % (configuration.save_dir))
            file_name_weight = utils.GetLastWeight(configuration.save_dir)
            print("The weight file %s will be used..." %(file_name_weight))
        else:
            file_name_weight = self.args.weights
        json = model.to_json()
        import datetime

        #currentDT = datetime.datetime.now()
        #currentDT.strftime("%Y-%m-%d %H:%M:%S")
        with open('./%s/model.json' %(configuration.save_dir,configuration.model_name), 'w') as f:
            f.write(json)
        
        #print("Loading from json")
        from keras.models import model_from_json
        model = model_from_json(json, custom_objects={'RotationThetaWeightLayer': utils.RotationThetaWeightLayer, "JunctionWeightLayer": utils.JunctionWeightLayer})
        model.summary()
        #plot_model(model, to_file=configuration.save_dir+'/model.png', show_shapes=True)
        
        if not self.args.testing:
            if (configuration.parallel):
                multi_model = multi_gpu_model(model, gpus=configuration.gpu)
                if file_name_weight is not None:
                    from keras.models import load_model 
                
                    #multi_model = load_model(file_name_weight, custom_objects={'dce_c':metrics.dce_c, 'RotationThetaWeightLayer': utils.RotationThetaWeightLayer, 'JunctionWeightLayer': utils.JunctionWeightLayer})
                    multi_model.load_weights(file_name_weight)
                else:
                    print("No weight was loaded...")
                self.Train(model=multi_model)
            else:
                if file_name_weight is not None:
                    #model = load_model(file_name_weight, custom_objects={'dce_c':metrics.dce_c, 'RotationThetaWeightLayer': utils.RotationThetaWeightLayer, 'JunctionWeightLayer': utils.JunctionWeightLayer})
                    
                    model.load_weights(file_name_weight)
                else:
                    print("No weight was loaded...")
                self.Train(model=model)
            print("Saving the model...")
            model.save(configuration.save_dir + '/trained_model.h5')
            print('Trained model saved to \'%s/trained_model.h5\'' % configuration.save_dir)
            self.Test(model=multi_model, args=args, type_data="None", color_mode=configuration.color_mode)
        else:  # as long as weights are given, will run testing
            if (configuration.parallel):
                multi_model = multi_gpu_model(model, gpus=configuration.gpu)
            
                if file_name_weight is None:
                    print('No weights are provided. Random weight will be initiated...')
                else:
                    print("Load the weights from the file:", file_name_weight)
                    multi_model.load_weights(file_name_weight, by_name=True)
                
                self.Test(model=multi_model, args=args, type_data=configuration.type_data_for_test, color_mode=configuration.color_mode)
            else:

                if file_name_weight is None:
                    print('No weights are provided. Random weight will be initiated...')
                else:
                    print("Load the weights from the file:", file_name_weight)
                    model.load_weights(file_name_weight, by_name=True)
                
                self.Test(model=model, args=args, type_data=configuration.type_data_for_test, color_mode=configuration.color_mode)

    def Train(self, model):
        print('-'*30 + 'Begin: training ' + '-'*30)
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        
        #Train
        #Callbacks
        print("Proc: Preprare the callbacks...")
        log = callbacks.CSVLogger(configuration.save_dir + '/log.csv')
        tb = callbacks.TensorBoard(log_dir=configuration.save_dir + '/tensorboard-logs',
                               batch_size=configuration.batch_size, histogram_freq=self.args.debug)

        callback=[log, tb] + configuration.list_for_callbacks
        print("Done: callbacks are created...")
        # compile the model 
        print("Proc: Compile the model...")
        model.compile(optimizer=configuration.optimizer,
                        loss_weights=configuration.loss_weights,
                        loss=configuration.loss, #self.loss_sigmoid_cross_entropy_with_logits,#util.margin_loss,
                        metrics=configuration.metrics)
        print("Done: the model was complied...")


        print("Proc: Training the model...")
        # Training with data augmentation
        total_number = int(round(len(configuration.training_set_index)*configuration.augmentation_factor_for_training))
        train_steps_per_epoch = int(round((total_number// configuration.batch_size)))
        valid_steps_per_epoch = (len(configuration.validation_set_index)*configuration.augmentation_factor_for_validation) // configuration.batch_size
        test_steps_per_epoch = (len(configuration.test_set_index)) // configuration.batch_size
        
        print('train_steps_per_epoch',train_steps_per_epoch)
        print('valid_steps_per_epoch',valid_steps_per_epoch)
        if configuration.test_mode:
            for x,y in utils.DataGenerator(list_IDs=configuration.training_set_index,
                                                hdf5_path=configuration.dataset_hdf5_path,
                                                batch_size=configuration.batch_size,
                                                dim=configuration.input_shape,
                                                n_channels=configuration.number_input_channel,
                                                n_classes=configuration.n_class,
                                                shuffle=configuration.shuffle,
                                                run_augmentations=configuration.run_augmentations,
                                                mode="training",
                                                convert_to_categorical=configuration.convert_to_categorical,
                                                binarize=configuration.binarize,
                                                threshold_to_binary=configuration.threshold_to_binary,
                                                Normalization=configuration.Normalization,
                                                Two_output=configuration.Two_output):
                print(x.shape)
                print('max', np.max(x))
                print('min', np.min(x))
                print('mean', np.mean(x))
                print('median', np.median(x))
                print(y[0].shape)
                print(y[1].shape)
        else:
            train_generator = utils.DataGenerator(list_IDs=configuration.training_set_index,
                                                hdf5_path=configuration.dataset_hdf5_path,
                                                batch_size=configuration.batch_size,
                                                dim=configuration.input_shape,
                                                n_channels=configuration.number_input_channel,
                                                n_classes=configuration.n_class,
                                                shuffle=configuration.shuffle,
                                                run_augmentations=configuration.run_augmentations,
                                                mode="training",
                                                convert_to_categorical=configuration.convert_to_categorical,
                                                binarize=configuration.binarize,
                                                threshold_to_binary=configuration.threshold_to_binary,
                                                Normalization=configuration.Normalization,
                                                Two_output=configuration.Two_output)

            validation_input_generator = utils.DataGenerator(list_IDs=configuration.validation_set_index,
                                                hdf5_path=configuration.dataset_hdf5_path,
                                                batch_size=configuration.batch_size,
                                                dim=configuration.input_shape,
                                                n_channels=configuration.number_input_channel,
                                                n_classes=configuration.n_class,
                                                shuffle=configuration.shuffle,
                                                run_augmentations=False,
                                                mode="training",
                                                convert_to_categorical=configuration.convert_to_categorical,
                                                binarize=configuration.binarize,
                                                threshold_to_binary=configuration.threshold_to_binary,
                                                Normalization=configuration.Normalization,
                                                Two_output=configuration.Two_output)

            test_input_generator = utils.DataGenerator(list_IDs=configuration.test_set_index,
                                                hdf5_path=configuration.dataset_hdf5_path,
                                                batch_size=configuration.batch_size,
                                                dim=configuration.input_shape,
                                                n_channels=configuration.number_input_channel,
                                                n_classes=configuration.n_class,
                                                shuffle=configuration.shuffle,
                                                run_augmentations=False,
                                                mode="prediction",
                                                convert_to_categorical=configuration.convert_to_categorical,
                                                binarize=configuration.binarize,
                                                threshold_to_binary=configuration.threshold_to_binary,
                                                Normalization=configuration.Normalization,
                                                Two_output=configuration.Two_output)

            model.fit_generator(train_generator,
                            steps_per_epoch=train_steps_per_epoch,
                            epochs=configuration.epochs,
                            use_multiprocessing=configuration.use_multiprocessing,
                            max_queue_size=configuration.max_queue_size,
                            workers=configuration.workers,
                            class_weight=configuration.class_weight,
                            validation_steps=valid_steps_per_epoch,
                            validation_data=validation_input_generator,
                            callbacks=callback)

            print("Run: test phase")
            print('test_steps_per_epoch',test_steps_per_epoch)
            print("Number of test cases", len(configuration.test_set_index))
            y_predict=model.predict_generator(test_input_generator,
                            steps_per_epoch=train_steps_per_epoch,
                            epochs=configuration.epochs,
                            use_multiprocessing=configuration.use_multiprocessing,
                            max_queue_size=configuration.max_queue_size,
                            workers=configuration.workers,
                            class_weight=configuration.class_weight,
                            validation_steps=valid_steps_per_epoch,
                            validation_data=validation_input_generator,
                            callbacks=None)
            y_true_ = keras.utils.HDF5Matrix(configuration.dataset_hdf5_path, 'label')
            y_true = np.asarray(y_true_>=configuration.threshold_to_binary, dtype=np.uint8)
            from sklearn.metrics import auc,average_precision_score, precision_recall_curve,classification_report, f1_score, confusion_matrix,brier_score_loss
            from sklearn.metrics import roc_auc_score,roc_curve ,fowlkes_mallows_score

            title="ROC Curve for case detection with prostate cancer"
            utils.plotROCCurveMultiCall(plt,y_true, y_predict, title)
            plt.savefig("%s/PCA_MRI_DETECTION_roc_curve.eps"%(configuration.save_dir), transparent=True)
            plt.savefig("%s/PCA_MRI_DETECTION_roc_curve.pdf"%(configuration.save_dir), transparent=True)
            plt.savefig("%s/PCA_MRI_DETECTION_roc_curve.png"%(configuration.save_dir), transparent=True)
            plt.show()
            plt.close()
            fpr, tpr, threshold = roc_curve(y_true, y_predict)
            roc_auc = auc(fpr, tpr)
            print('roc_auc',roc_auc)
            threshold =utils.cutoff_youdens(fpr, tpr, threshold)
            print('threshold',threshold)
            print("Confusion matrix")
            print(confusion_matrix(y_true, y_predict>threshold))
            print("Classification report")
            print(classification_report(y_true, y_predict>threshold))
            print("fowlkes_mallows_score")
            fms = fowlkes_mallows_score(y_true, y_predict>threshold)
            print(fms)
            brier_score =brier_score_loss(y_true, y_predict)
            print("brier score")
            print(brier_score)
            print("END: TESTING THE MODEL")
 

        print('-'*30 + 'End: training ' + '-'*30)

    def Test(self, model, args, type_data="folder", color_mode="RGB"):
        '''
        Not implemented
        '''
        if type_data == "folder":
            files = [f for f in os.listdir(args.path_test) if os.path.isfile(os.path.join(args.path_test, f)) and os.path.splitext(f)[1] in [".tiff", ".tif",".svs"]]
            
            for file_name in files:
                if file_name.startswith(".") == False:
                    file_ls = os.path.join(args.path_test,file_name)
                    print("file:",file_ls)
                    pass
        elif type_data == "file":
            pass

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Cancer Detector on histology images.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
          help="argparseThe path of the saved weights. Should be specified when testing")
    parser.add_argument('-p', '--parallel', default=False, action='store_true',
                        help="Set parallel computation.")
    parser.add_argument('-lw', '--load_previous_weight', default=False, action='store_true',
                        help="Load a previous trained weight.")
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help="Show hidden messages.")
    args = parser.parse_args()
    
    print(args)

    if not os.path.exists(configuration.save_dir):
        os.makedirs(configuration.save_dir)

    Prostate3DMODEL_CLASSIFICATION = ModelRunner(args)
    Prostate3DMODEL_CLASSIFICATION.Run()