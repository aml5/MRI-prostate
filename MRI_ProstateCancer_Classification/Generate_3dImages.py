import keras
import SimpleITK as sitk
import random
import DataGenerator
import h5py
import numpy as np
import scipy
from mayavi import mlab
from scipy import ndimage
import matplotlib.pyplot as plt
import keras.backend as K
from skimage.exposure import equalize_adapthist

def Normalize(data, normalize=None):
    if normalize=='CLAHE':
        ct_scan = D3_equalize_adapthist(ct_scan, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
    elif normalize=='CLAHE2':
        ct_scan /= np.max(ct_scan)
        ct_scan = D3_equalize_adapthist(ct_scan, clip_limit=configuration.clip_limit, nbins=configuration.nbins)
    elif normalize=='ZSCORE':
        ct_scan = (ct_scan - np.mean(ct_scan))/(np.std(ct_scan))
    elif normalize=='MAX':
        ct_scan = ct_scan / np.max(ct_scan)
    elif normalize=="tanh":
        X_train_pos_ = data
        vle_min = np.min(X_train_pos_)
        vle_max = np.max(X_train_pos_)
        for index in range(X_train_pos_.shape[0]):
            X_train_pos = X_train_pos_[index]
            mean_ = np.mean(X_train_pos)
            std_ = np.std(X_train_pos)
            data_ = np.tanh((X_train_pos-mean_)/std_) 
            X_train_pos_[index] = data_
        ct_scan = X_train_pos_
    else:
        ct_scan = ct_scan
    return ct_scan

def ApplyAugmentation(d3_img, type_of_augmentation=None, dict_parameter=None):
    if dict_parameter is None:
        dict_parameter={'rotation_xy':[0,4],
                        'rotation_zx' :[1,4],
                        'rotation_zy' :[1,4],
                        'zooming':[1.05,1.15],
                        #'down_scale':[1,0.8]
                        }
    if type_of_augmentation is None:
        seq=[
            'rotation_xy',
            'rotation_zx',
            'rotation_zy',
            'zooming', 
            #'down_scale',
            #'h_flip',
            #'v_flip',
            #'z_flip',
            #'rotate_90_k1',
            #'rotate_90_k2',
            #'rotate_90_k3'
            ]
        type_of_augmentation = random.choice(seq)
    if type_of_augmentation=='rotation_xy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(1,2),reshape=False)
    elif type_of_augmentation=='rotation_zx':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(0,2),reshape=False)
    elif type_of_augmentation=='rotation_zy':
        angle = random.randint(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_3d_img = scipy.ndimage.rotate(d3_img, angle, axes=(0,1),reshape=False)
    elif type_of_augmentation=='zooming':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img, (1, value_factor, value_factor))
        x_a = new_img_zoom.shape[2]//2 - 72
        y_a = new_img_zoom.shape[1]//2 - 72
        new_3d_img = new_img_zoom[:,y_a:y_a+144, x_a:x_a+144]
    elif type_of_augmentation=='down_scale':
        value_factor = random.uniform(dict_parameter[type_of_augmentation][0],dict_parameter[type_of_augmentation][1])
        new_img_zoom =  ndimage.zoom(d3_img, (1, value_factor, value_factor))
        new_img_zoom_tmp = np.zeros_like(d3_img)
        x_a = new_img_zoom.shape[2]//2
        y_a = new_img_zoom.shape[1]//2
        x_a_b = new_img_zoom_tmp.shape[2]//2 - x_a
        y_a_b = new_img_zoom_tmp.shape[1]//2 - y_a
        new_img_zoom_tmp[:,y_a_b:y_a_b+new_img_zoom.shape[1], x_a_b:x_a_b+new_img_zoom.shape[1]] = new_img_zoom
        new_3d_img = new_img_zoom_tmp.copy()
    elif type_of_augmentation == 'h_flip':
        new_3d_img = np.flip(d3_img,axis=1)
    elif type_of_augmentation == 'v_flip':
        new_3d_img = np.flip(d3_img,axis=2)
    elif type_of_augmentation == 'z_flip':
        new_3d_img = np.flip(d3_img,axis=0)
    elif type_of_augmentation=='rotate_90_k1':
        new_3d_img = np.rot90(d3_img,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k2':
        new_3d_img = np.rot90(d3_img,k=2,axes=(1,2))
    elif type_of_augmentation=='rotate_90_k3':
        new_3d_img = np.rot90(d3_img,k=3,axes=(1,2))
    else:
        new_3d_img = d3_img
    return new_3d_img.copy()

def GenerateNumpyData(h5_file, type_mpMRI, selected_plts, CoRegistration, standard_volume=[20,384,384], patch_size=(144,144), z_range=(2,18), mask=None):
    mri_img_c=[]
    patient_list = {}
    lesions_info = DataGenerator.get_lesion_info(h5_file, type_mpMRI, selected_plts, CoRegistration=None)
    for lesions_info_ in lesions_info:
        #Resize all MRI to the same size.
        img = lesions_info_[1]
        new_x_size = standard_volume[1]
        new_y_size = standard_volume[2]
        new_z_size = standard_volume[0]
        img = sitk.GetImageFromArray(img)
        new_size = [new_x_size, new_y_size, new_z_size]
        new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), new_size)]
        interpolator_type = sitk.sitkLinear
        new_img = sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, img.GetPixelIDValue())
        x_0 = new_x_size//2 -patch_size[0]//2
        y_0 = new_y_size//2 -patch_size[1]//2
        y_1 = y_0 + patch_size[1]
        x_1 = x_0 + patch_size[0]

        new_img_ = sitk.GetArrayFromImage(new_img)
        new_img_ = new_img_[z_range[0]:z_range[1], y_0:y_1, x_0:x_1]
        #print(np.mean(new_img_))
        #print(np.min(new_img_))
        #print(np.max(new_img_))
        mri_img_c.append(new_img_.copy())

        for lesion in lesions_info_[0]:
            name = lesion['name'].split('/')[1]
            Sig = str(lesion['ClinSig'].decode("utf-8"))
            if Sig=='True':
                Sig=True
            else:
                Sig=False
            if name in patient_list:
                if patient_list[name]!=Sig:
                    if patient_list[name]==False and Sig==True:
                        patient_list[name]==True
            else:
                patient_list[name] = Sig
    print('patient number:', len(patient_list.keys()))
    i = i = sum(1 if b else 0 for b in patient_list.values())
    print('positive',i)
    print('negative', len(patient_list.keys())-i)
    return list(patient_list.values()), mri_img_c

def GenerateDatasetWith3DImage(h5_file, folder_to_store='./', type_of_mpMRI=['t2_tse_tra'], train_portion=0.8, standard_volume=[20,384,384], patch_size=(144,144), z_range=(2,18), mask=None):
    ptl_list = list(h5_file.keys())
    select_nr = int(round(len(ptl_list)*train_portion))
    print('#'*30)
    print('Training set:')
    print('n=',select_nr)
    #Random selection...
    selected_plts=random.sample(ptl_list,k=select_nr)
    print(selected_plts)
    print('#'*30)
    print('Valid set:')
    print('n=',len(ptl_list)-select_nr)
    remaining = DataGenerator.difference_btn_twolist(ptl_list, selected_plts)
    print(remaining)
    print('#'*30)
    #Training set:
    print("Generating training set...")
    for type_MRI in type_of_mpMRI:
        patient_list, mri_img_c = GenerateNumpyData(h5_file, 
                                                    type_MRI, 
                                                    selected_plts, 
                                                    None, 
                                                    standard_volume=standard_volume, 
                                                    patch_size=patch_size, 
                                                    z_range=z_range, 
                                                    mask=mask
                                                    )
        file_img = folder_to_store + 'img_train_data_3d_%s.npy' % (type_MRI)
        file_outcome = folder_to_store + 'outcome_train_data_3d_%s.npy' % (type_MRI)
        print("Augmentation for validations set...")
        GenerateAugmentationAndStoreThem(mri_img_c, patient_list, file_img=file_img, file_outcome=file_outcome, seq_c=['None', 'rotation_xy', 'rotation_zx', 'rotation_zy', 'zooming'])
    #Validation set:
    print("Generating validations set...")
    for type_MRI in type_of_mpMRI:
        patient_list, mri_img_c =GenerateNumpyData(h5_file,
                                                    type_MRI, 
                                                    remaining, 
                                                    None, 
                                                    standard_volume=standard_volume, 
                                                    patch_size=patch_size, 
                                                    z_range=z_range, 
                                                    mask=mask
                                                    )
        file_img = folder_to_store + 'img_valid_data_3d_%s.npy' % (type_MRI)
        file_outcome = folder_to_store + 'outcome_valid_data_3d_%s.npy' % (type_MRI)
        print("Augmentation for validations set...")
        GenerateAugmentationAndStoreThem(mri_img_c, patient_list, file_img=file_img, file_outcome=file_outcome, seq_c=['None'])
    print('Proc: Done')

def GenerateAugmentationAndStoreThem(numpy_img_list, 
                                    numpy_outcome_list, 
                                    patch_volume=(16,144,144,1), 
                                    file_img='./img_valid_data_3d.npy', 
                                    file_outcome='./outcome_valid_data_3d.npy',
                                    seq_c =['None', 'rotation_xy', 'rotation_zx', 'rotation_zy', 'zooming'],
                                    verbose=False):
    augmented=[]
    out_come_aug = []
    #seq_c =['None', 'rotation_xy', 'rotation_zx', 'rotation_zy', 'zooming', 
    #    'down_scale', 'h_flip', 'v_flip', 'z_flip', 'rotate_90_k1',
    #    'rotate_90_k2', 'rotate_90_k3']
    #'down_scale']
    for i in range(len(numpy_outcome_list)):
        for seq in seq_c:
            if verbose:
                print('Runing Augmentation: %s' % seq)
            img_c = ApplyAugmentation(numpy_img_list[i], type_of_augmentation=seq)

            seq_w = seq_c.copy()
            seq_w.remove(seq)
            if len(seq_w)==0:
                augmented.append(img_c)
                out_come_aug.append(numpy_outcome_list[i])
            else:
                for seq_x in seq_w:
                    if verbose:
                        print('Runing A second augmentation: %s' % seq_x)
                    img_c_ = ApplyAugmentation(img_c, type_of_augmentation=seq_x)
                    augmented.append(img_c_)
                    out_come_aug.append(numpy_outcome_list[i])

    #augmented = numpy_img_list + augmented
    #out_come_aug = numpy_outcome_list + out_come_aug
    
    mri_img_c = np.zeros((len(out_come_aug), patch_volume[0], patch_volume[1], patch_volume[2], patch_volume[3]), dtype=K.floatx())
    for i in range(len(out_come_aug)):
        mri_img_c[i] = augmented[i].reshape(patch_volume).copy()
    out_come_aug = np.array(out_come_aug)
    print('Shape: ',mri_img_c.shape)
    print('Shape for outcome:', out_come_aug.shape)
    print('Save the result of the image augmentation and outcome...')
    np.save(file_img, mri_img_c)
    np.save(file_outcome, out_come_aug)

if __name__ == "__main__":    
    with h5py.File('./database.h5','a') as f:
        GenerateDatasetWith3DImage(f)
