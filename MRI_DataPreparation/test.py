# %
with open("AS_blocklist.csv") as f:
    block_id = f.read().lower().splitlines()
print(block_id)
# %
import json

with open(".\Data\StudyCohort.json") as f:
    json_ = f.read()
    StudyCohort = json.loads(json_)
# %
# n = normal, b = AS, wm an c := RPE
data = {}
data['category'] = []
for i, itm in enumerate(StudyCohort):
    AccessionID = itm['AccessionID'].lower()
    if "n" in AccessionID:
        data['category'].append("NORM")
        StudyCohort[i]['category'] = "NORM"
    elif "c" in AccessionID:
        data['category'].append("sPCA")
        StudyCohort[i]['category'] = "NORM"
    elif "wm" in AccessionID:
        data['category'].append("sPCA")
        StudyCohort[i]['category'] = "NORM"
    elif "b" in AccessionID:
        if AccessionID not in block_id:
            data['category'].append("iPCA")
            StudyCohort[i]['category'] = "iPCA"
        else:
            data['category'].append("sPCA")
            StudyCohort[i]['category'] = "sPCA"
# %


from collections import Counter

print(Counter(data['category']))

# %
import numpy as np

image_types = []
for x in StudyCohort:
    image_types.extend(x['data']['ImageType'])
# %
myset = set(image_types)
myset
#

myset = set(image_types)
'''
 'PRIMARY_CMB', => Bleeding


 'PRIMARY_IP',

 'PRIMARY_MAVRIC',
 'PRIMARY_M_IR',
 'PRIMARY_PROJECTION IMAGE',
 'PRIMARY_PROPELLER',
 'PRIMARY_W'}
 '''
# T2 images => "PRIMARY_M_SE", "PRIMARY_M_FFE", 'PRIMARY_OTHER', 'PRIMARY_M','PRIMARY_T2', 'PRIMARY_AXIAL'
# ADC images => 'PRIMARY_ADC','PRIMARY_ADC_UNSPECIFIED','PRIMARY_EADC','PRIMARY_EADC_UNSPECIFIED', 
# Diffusion Tensor Imaging  => 'PRIMARY_FA'
# PRIMARY_DIFFUSION =>PRIMARY_DIFFUSION
# functional MRI => 'PRIMARY_F'
# 'PRIMARY_DIXON' => Water => DWI

# %
img_type = [x['data']['ImageType'] for x in StudyCohort]
img_type_lst = np.array(img_type).flatten()
print(np.unique(img_type_lst))
# %
import pydicom
import os
import matplotlib.pyplot as plt

# source_dir = r"C:\Users\Okyaz Eminaga\Documents\16edd4e-c346\16edd4e-c346\\1.2.840.4267.32.314766340130186156553672218843842311973"
# source_dir = r"C:\Users\Andrew Lu\Documents\Projects\MRI\dcm"
# directory_ = [os.path.join(source_dir, x) for x in os.listdir(source_dir)]
#
# for subdir in directory_:
#     file_ = os.listdir(subdir)[0]
#     print(file_)
#     dim = pydicom.read_file(os.path.join(subdir, file_))
#     print(dim.ImageType)
# # %
# dim = pydicom.read_file(
#     r"C:\Users\Okyaz Eminaga\Documents\16edd4e-c346\16edd4e-c346\1.2.840.4267.32.314766340130186156553672218843842311973\1.2.840.4267.32.98977683778372994933979599229438123379\000004-1.2.840.4267.32.85551342503812153255755582679412945747.dcm")
# print(dim)
# # %
# ['1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.319416895985609821808077489710001832091',
#  '1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.102037670313200109613853380336333946977',
#  '1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.292353590962169623329991790897897196009',
#  '1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.196962239951551989614223106169319352881',
#  '1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.70716200353438902791196940316312406100',
#  '1.2.840.4267.32.305620313887178705286425764153618050136/1.2.840.4267.32.252274516192297662067817469928650197011']
# %
# 'ImageType': ['PRIMARY_DIXON',
#               'PRIMARY_OTHER',
#               'PRIMARY_T2',
#               'PRIMARY_ADC',
#               'PRIMARY_EADC',
#               'PRIMARY_CMB']


# %
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

                if (R_ == 1 and F_ == 1):
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

    # Select only the most recent processed serie number
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
                            type_of_data[index]['number_img'] < 70 and type_of_data[index]['number_img'] > 7):
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
    if len(data['AccessionID']) == 0:
        print(data)
        return None

    item['filename'] = filename
    item['data'] = data
    item['AccessionID'] = data['AccessionID'][0]

    return item
