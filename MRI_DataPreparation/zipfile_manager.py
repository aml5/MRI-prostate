##############
#
# Copyright by Okyaz Eminaga
#
##############
from tarfile import TarFile
import tarfile
import os
import json
import pydicom
##############
#
#   Define the location of the zip file
#
##############
#%% Generate the file list
FolderThatContainsTheFiles= r'./data/tgz' #r"Z:\MRI_PRAD\MRI_PRAD"
files = [os.path.join(FolderThatContainsTheFiles, x) for x in os.listdir(FolderThatContainsTheFiles) if x.endswith('.tgz')]
# %% Extract the info required for data
def GetAndStoreInfo(filename):
    files_dictionary = {}
    type_of_data = []
    f: TarFile
    with tarfile.open(filename) as f:
        members = f.getmembers()
        for mem in members:
            if mem.isfile():
                path, file = os.path.split(mem.name)
                f_f = f.extractfile(mem)
                dcim = pydicom.read_file(f_f)
                R_, L_, A_, P_, F_, H_ = dcim.ImageOrientationPatient

                if (R_>0.90 and F_ >0.90) and (L_<0.05 and A_<0.05):
                    if path not in files_dictionary:
                        files_dictionary[path] = []
                        info_ = {'unique': path, 'ImageOrientation': dcim.ImageOrientationPatient,
                                 'SeriesDescription': dcim.SeriesDescription,
                                 'Series Number': dcim.SeriesNumber, 'AccessionNumber': dcim.AccessionNumber,
                                 'ImageType': dcim.ImageType, 'Date': dcim.StudyDate}
                        type_of_data.append(info_)
                    files_dictionary[path].append(mem.name)
    for index, itm in enumerate(type_of_data):
        type_of_data[index]['number_img'] = len(files_dictionary[itm['unique']])
        type_of_data[index]['file_list'] = files_dictionary[itm['unique']]
    sr = [val['ImageType'][1] + "_" + val['ImageType'][2] for val in type_of_data]

    s = set(sr)

    # Select only the most recent processed series number
    get_max_serie_number = {}
    for index, key in enumerate(sr):
        for unique_key in s:
            if key == unique_key:
                if key not in get_max_serie_number:
                    get_max_serie_number[key] = type_of_data[index]['Series Number']
                else:
                    recent_value = get_max_serie_number[key]
                    current_value = type_of_data[index]['Series Number']
                    if recent_value < current_value and (
                            type_of_data[index]['number_img'] < 80 and type_of_data[index]['number_img'] > 7):
                        get_max_serie_number[key] = current_value

    # Save using the Accession id, foldername, the file list and image type.date, age.
    data = {}
    data['AccessionID'] = []
    data['Folder'] = []
    data['FileList'] = []
    data['Date'] = []
    data['ImageType'] = []
    data['SeriesNumber'] = []
    serie_number_list = [x for x in get_max_serie_number.values()]

    for index, itm in enumerate(type_of_data):
        if itm['Series Number'] in serie_number_list:
            data['AccessionID'].append(itm['AccessionNumber'])
            data['Folder'].append(itm['unique'])
            data['FileList'].append(itm['file_list'])
            data['Date'].append((itm['Date']))
            data['ImageType'].append(sr[index])
            data['SeriesNumber'].append(itm['Series Number'])

    item = {}
    if len(data['AccessionID'])==0:
        print(data)
        return None

    item['filename'] = filename
    item['data'] = data
    item['AccessionID'] = data['AccessionID'][0]

    return item
#%%

Study_coordinator = []
print("INFO: ", len(files), ' files were found...')
for filename in files:
    print("PROC:", filename)
    item = GetAndStoreInfo(filename)
    if item is None:
        print("ERR:",filename)

        continue
    Study_coordinator.append(item.copy())
    #p
    # print("DONE:", filename)
print("PROC: storing the information into /data/StudyCohort3.json")
with open('./data/StudyCohort3.json', 'w') as outfile:
    json.dump(Study_coordinator, outfile)
print("Done: storing the information into /data/StudyCohort3.json")
print("FINISHED: LEAVING THE SCRIPT")