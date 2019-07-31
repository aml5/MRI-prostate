import SimpleITK as sitk
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import matplotlib.gridspec as gridspec
# import cv2
import os
from os.path import isfile, join
from os import listdir
import argparse
import keras
from keras import optimizers, callbacks, models
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import loss_functions
import model_storage
import cv2
from PIL import Image, ImageEnhance
import PIL
from skimage import measure
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import exposure
from sklearn.preprocessing import QuantileTransformer
from skimage.morphology import remove_small_objects
import configuration
import progressbar
import time
import D3_utils as utils
import tensorflow as tf
import model_storage
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= configuration.initial_gpu
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

###########
#
#   Training/Testing
#
###########
def Run(args):
    #Define the model
    h_pars = configuration.hyperparameters[configuration.select_model]

    loader = model_storage.Model_Storage(input_shape=h_pars['input_shape'],
                                        n_filter=h_pars['n_filter'], 
                                        number_of_class=h_pars['number_of_class'],
                                        activation_last=h_pars['activation_last'],
                                        metrics=h_pars['metrics'],
                                        loss=h_pars['loss'],
                                        optimizer=h_pars['optimizer']
                                        )
    
    #Check if the folder exists for the result and weight storage
    timestr = time.strftime("%Y%m%d-%H%M")
    to_store_result = './results/result_' + (configuration.select_model) + "_" + (timestr)
    configuration.hyperparameters[configuration.select_model]['to_save'] = to_store_result

    if os.path.exists(to_store_result):
        pass
    else:
        os.mkdir(to_store_result)
    
    #Load the model
    if (configuration.parallel):
        with tf.device('/cpu:0'):
            method_to_call = getattr(loader, configuration.select_model) #'CancerDetection_HarmonicSeries')
            model = method_to_call()
    else:
        method_to_call = getattr(loader, configuration.select_model) #'CancerDetection_HarmonicSeries')
        model = method_to_call()   
    print(model.summary())

    from keras.models import model_from_json, load_model
    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    if args.load_model:

        print("Proc: Loading the previous model...")
        # load json and create model
        loaded_model_json = None
        with open(to_store_result + '/trained_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        #Load the weight is available...
    if args.weights is None and args.last_weight:
        file_name_weight = utils.GetLastWeight(to_store_result)
    else:
        file_name_weight = args.weights

    if file_name_weight is not None:
        print('Loading the weight stored in the file %s' % (file_name_weight))
        model.load_weights(file_name_weight, by_name=True)
    else:
        print('No weights are provided. Will test using random initialized weights.')
    
    if configuration.parallel:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=configuration.number_of_gpus, cpu_relocation=False, cpu_merge=True)
        model = loader.Compile(model)
    else:
        #Compile the model.
        model = loader.Compile(model)

    if not args.testing:
        #plot_model(model, to_file=to_store_result+'/model.png', show_shapes=True)
        train(args, model)  
        file_name_weight = utils.GetLastWeight(to_store_result)
        if file_name_weight is not None:    
            model.load_weights(file_name_weight, by_name=True)
        # test(args, model, verbose=args.verbose)
    else:
        test(args, model=model, verbose=args.verbose)
 
def train(args, model, verbose=False):
    valid_img_dataset = np.load(configuration.img_data_valid)
    valid_mask_dataset = np.load(configuration.mask_data_valid)
    train_img_dataset = np.load(configuration.img_data_train)
    train_mask_dataset = np.load(configuration.mask_data_train)
    
    print("training set: ",train_img_dataset.shape)
    print("training set -mask: ",train_mask_dataset.shape)
    print("validation set: ",valid_img_dataset.shape)
    print("validation set -mask: ",valid_mask_dataset.shape)
    '''
    selected_randomly_to_fill_the_gap = np.random.choice(train_img_dataset.shape[0], 10)
    '''
    #train_img_dataset = train_img_dataset - 0.485
    #valid_img_dataset = valid_img_dataset - 0.485
    '''
    print('train')
    print('mean',np.mean(train_img_dataset))
    print('median',np.median(train_img_dataset))
    print('min',np.min(train_img_dataset))
    print('max',np.max(train_img_dataset))
    print('valid')
    print('mean',np.mean(valid_img_dataset))
    print('median',np.median(valid_img_dataset))
    print('min',np.min(valid_img_dataset))
    print('max',np.max(valid_img_dataset))
    '''
    # callbacks
    print("Proc: Preprare the callbacks...")
    save_dir = configuration.hyperparameters[configuration.select_model]['to_save']
    batch_size =  configuration.hyperparameters[configuration.select_model]['batch_size']
    print("Save_dir: " + save_dir)
    log = callbacks.CSVLogger(save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=save_dir + '/tensorboard-logs',
                                batch_size=batch_size, histogram_freq=args.debug)
    monitor_par = configuration.hyperparameters[configuration.select_model]['monitor']
    mode_par = configuration.hyperparameters[configuration.select_model]['mode']
    checkpoint = callbacks.ModelCheckpoint(save_dir + '/weights-{epoch:02d}.h5', monitor=monitor_par,
                                            save_best_only=True, save_weights_only=True, mode=mode_par,verbose=1)
    print("Done: callbacks are created...")
    print("Proc: Model loading...")
    train_sample_size = configuration.hyperparameters[configuration.select_model]['train_sample_size']
    valid_sample_size = configuration.hyperparameters[configuration.select_model]['valid_sample_size']
    epochs = args.epochs

    train_steps_per_epoch = train_sample_size // batch_size
    valid_steps_per_epoch = 1 #valid_sample_size // batch_size

    call_backs = [log, tb, checkpoint] + configuration.hyperparameters[configuration.select_model]['AdditionalCallbacks']
    print("Proc: training...")
    '''
    for (x,y) in utils.DataGeneratorWithAugmentation(X=train_img_dataset,Y=train_mask_dataset,batch_size=batch_size, 
                        RunNormalize=configuration.hyperparameters[configuration.select_model]['RunNormalize']):
        print(x.shape)
        print(y.shape)
        for x_n in x:
            for x_c in x_n:
                plt.imshow(x_c[...,0])
                plt.show()
    
    '''
    print('train_steps_per_epoch',train_steps_per_epoch)
    print('valid_steps_per_epoch',valid_steps_per_epoch)
    # blurred1 = cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 1.0)
    # blurred2 = cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 15.0)
    # blurred3 = cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 225.0)
    # blurred4 = cv2.bilateralFilter(np.float32(train_img_dataset[6][5].reshape(384, 384)), 9, 25.0, 25.0)
    # plt.subplot(221), plt.imshow(blurred1), plt.title('sigma 1.0')
    # plt.subplot(222), plt.imshow(blurred2), plt.title('sigma 15.0')
    # plt.subplot(223), plt.imshow(blurred3), plt.title('sigma 225.0')
    # plt.subplot(224), plt.imshow(blurred4), plt.title('bilateral 9')
    # plt.show()
    # img = train_img_dataset[6][5].reshape(384, 384)
    # plt.subplot(131),plt.imshow(img)
    # plt.subplot(132),plt.imshow(utils.SharpTheImage(img,verbose=False,type_of_sharpness='Curvature'))
    # plt.subplot(133), plt.imshow(utils.SharpTheImage(img,verbose=False,type_of_sharpness='CV2'))
    # plt.show()
    # exit()
    # utils.sitk_show(imgSmooth)
    # blurred1 = utils.CorrectN4Bias(sitk.GetImageFromArray(cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 1.0)))
    # blurred2 = utils.CorrectN4Bias(sitk.GetImageFromArray(cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 15.0)))
    # blurred3 = utils.CorrectN4Bias(sitk.GetImageFromArray(cv2.GaussianBlur(train_img_dataset[6][5].reshape(384, 384), (9, 9), 225.0)))
    # blurred4 = utils.CorrectN4Bias(sitk.GetImageFromArray(cv2.bilateralFilter(np.float32(train_img_dataset[6][5].reshape(384, 384)), 9, 75, 75)))
    # plt.subplot(221), plt.imshow(blurred1), plt.title('sigma 1.0')
    # plt.subplot(222), plt.imshow(blurred2), plt.title('sigma 15.0')
    # plt.subplot(223), plt.imshow(blurred3), plt.title('sigma 225.0')
    # plt.subplot(224), plt.imshow(blurred4), plt.title('bilateral 9')
    # plt.show()
    # img = train_img_dataset[6][5].reshape(384, 384)
    # plt.subplot(121), plt.imshow(img)
    # img = sitk.GetArrayFromImage(utils.CorrectN4Bias(sitk.GetImageFromArray(img)))
    # plt.subplot(122), plt.imshow(img)
    # plt.show()
    # train_img_dataset = train_img_dataset.reshape(40, 24, 384, 384)
    if (args.s1):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Andrew1")
    elif (args.s15):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Andrew2")
    elif (args.s225):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Andrew3")
    elif (args.bilateral):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Bilateral")
    elif(args.curvature):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Curvature")
    elif(args.cv2):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="CV2")
    elif(args.sharp):
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i], verbose=False,type_of_sharpness="Classic")
    else:
        for i in range(train_img_dataset.shape[0]):
            train_img_dataset[i] = utils.Sharp3DVolume(train_img_dataset[i],verbose=False,type_of_sharpness=None)
    train_img_dataset = train_img_dataset.reshape(40, 24, 384, 384, 1)
    for volume in train_img_dataset:
        print(np.mean(volume))
        print(np.median(volume))
        print(np.std(volume))
        print('----------')

    model.fit_generator(generator=utils.DataGeneratorWithAugmentation(X=train_img_dataset,Y=train_mask_dataset,batch_size=batch_size, 
                        RunNormalize=configuration.hyperparameters[configuration.select_model]['RunNormalize'],
                        RunAugmentation=configuration.hyperparameters[configuration.select_model]['RunAugmentation'],HED=configuration.HED),
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        use_multiprocessing=configuration.use_multiprocessing,
                        validation_steps=valid_steps_per_epoch,
                        validation_data= (valid_img_dataset, [valid_mask_dataset.reshape(10,24,384,384,1)]*7),
                        #utils.DataGeneratorNative(X=valid_img_dataset,Y=valid_mask_dataset,batch_size=batch_size, 
                        #RunNormalize=configuration.hyperparameters[configuration.select_model]['RunNormalize']),
                        callbacks=call_backs)
    
    print("Done: training...")
        
def test(args, model, verbose=False):
    print('*****'*30)
    print('**'*10, 'Prc: test', '**'*10)
    print('*****'*30)
    print(args)
    print('Onlive',configuration.Onlive)
    print('Directory mode', configuration.directory_mode)
    Onlive=configuration.Onlive

    if Onlive:
        seg = utils.SegmentProstate(args.dicom, model)
        print('Proc: Saving segmentation result...')
        np.save(seg, './segmentation.npy')
    else:
        test_img_dataset = np.load(configuration.img_data_valid)
        test_mask_dataset = np.load(configuration.mask_data_valid)
        
        from sklearn.metrics import confusion_matrix,classification_report
        report_list = []
        report_list_one_hot = []
        print("Loading the weight...")
        model.load_weights(r"results/result_Prostate_D3_Segmentation_20190705-1805/weights-32.h5")#"C:\Users\Andrew Lu\Documents\Projects\MRI_Prostate_Segmentation\result_D3_Model_Segmentation\weights-41.h5")
        print("Running the prediction...")
        print(test_img_dataset.shape)
        results = model.predict(test_img_dataset)
        resutls = results[-1]
        print('resutls.shape', resutls.shape)
        with progressbar.ProgressBar(maxval=test_img_dataset.shape[0]) as bar:
            for i, itm in enumerate(zip(test_img_dataset,resutls)):

                img = itm[0]
                # heatmap_predicted_y_x = itm[1]
                heatmap_predicted_y_x = utils.smooth_contours(utils.smooth_contours(itm[1],type='CV2'))
                # print('heatmap_predicted_y_x.shape', heatmap_predicted_y_x.shape)
                mask = test_mask_dataset[i]
                img_ = img.reshape(1, img.shape[0], img.shape[1], img.shape[2],1)
                #heatmap_predicted_y_x = model.predict(img_)
                #print(heatmap_predicted_y_x.shape)
                utils.multi_slice_viewer(heatmap_predicted_y_x.reshape(configuration.standard_volume),mask.reshape(configuration.standard_volume),img.reshape(configuration.standard_volume))
                plt.show()
                mask_x = heatmap_predicted_y_x
                y_pred = mask_x.flatten()>0.5
                y_true = mask.flatten()>0.5
                dr = confusion_matrix(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)#, target_names=class_labels)

                if verbose:
                    print('Confusion Matrix only prostate tissue using x_x')
                    print(dr)
                    print("Between-step performance :")
                    print(report)

                if 'True' in report:
                    report_list_one_hot.append(report['True']['f1-score'])

                if verbose:
                    print('Confusion Matrix only prostate tissue')
                    dr = confusion_matrix(y_true, y_pred)
                    print(dr)
                    print("Between-step performance :")
                    print(report)
                if 'True' in report:
                    report_list.append(report['True']['f1-score'])

        savedir = configuration.hyperparameters[configuration.select_model]['to_save']
        utils.SaveTheResult(report_list_one_hot, savedir+'/report_f1score_all.pkl')
        print('F1 of prostate area, seg of prostate and contour')
        print('mean',np.mean(np.array(report_list_one_hot)))
        print('median',np.median(np.array(report_list_one_hot)))
        print('mean,[95% CI]:',utils.mean_confidence_interval(report_list_one_hot))
        utils.SaveTheResult(report_list, savedir+'/report.pkl')
        print('F1 of prostate area, only seg of prostate')
        print('mean',np.mean(np.array(report_list)))
        print('median',np.median(np.array(report_list)))
        print('mean,[95% CI]:',utils.mean_confidence_interval(report_list))
    print('*****'*30)
    print('**'*10, 'Done: test', '**'*10)
    print('*****'*30)
#####
# Geneerate numpy data for train and valid set.
#####
def gp():
    utils.GenerateDataSetSaveResults(configuration.data_path, ratio=configuration.train_valid_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom', help='define the path that include the dicom files', default='./test/')
    parser.add_argument("--test", help="the path where the data are stored",
                    default="./test/")         
    parser.add_argument("--save_dir", help="store the weights and tensorboard reports",
                    default="./result")
    parser.add_argument("--weights", help="load the weights",
                    default=None)
    parser.add_argument("--ratio", help="ratio of train set",
                    default=80, type=int)
    parser.add_argument("--lr_factor",
                    default=0.5, type=int)
    parser.add_argument("--min_lr",
                    default=0.00000000001, type=float)
    parser.add_argument("--lr",
                    default=0.0001, type=float)
    parser.add_argument("--change_lr_threshold",
                    default=2, type=int)
    parser.add_argument("--nb_class",
                    default=3, type=int)
    parser.add_argument("--batch_size", help="define the batch size, standard 16",
                    default=16, type=int)
    parser.add_argument("--epochs", help="define the batch size, standard 16",
                    default=configuration.hyperparameters[configuration.select_model]['epochs'], type=int)
    parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")
    parser.add_argument('-l', '--load_model', default=False, action='store_true',
                    help="Load a previous model.")
    parser.add_argument('-gp',"--path_generate", help="Run patch generation",
                    action="store_true")
    parser.add_argument('-lw',"--last_weight", help="load the last weight",
                    action="store_true")
    parser.add_argument('-la',"--load_numpy", help="load the numpy data",
                    action="store_true", default=True)
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('-t', '--testing', default=False, action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-p', '--parallel', default=False, action='store_true',
                        help="Run in parallel mode")
    parser.add_argument('--sharp', default=False, action='store_true',
                        help="Run sharp")
    parser.add_argument('--s1', default=False, action='store_true',
                        help="Run gaussian 1.0")
    parser.add_argument('--s15', default=False, action='store_true',
                        help="Run gaussian 15.0")
    parser.add_argument('--s225', default=False, action='store_true',
                        help="Run gaussian 225.0")
    parser.add_argument('--bilateral', default=False, action='store_true',
                        help="Run bilateral filter")
    parser.add_argument('--curvature', default=False, action='store_true',
                        help="Run curvature flow filter")
    parser.add_argument('--cv2', default=False, action='store_true',
                        help="Run OpenCV's filter")
    args = parser.parse_args()
    if args.path_generate:
        gp()
    else:
        Run(args)