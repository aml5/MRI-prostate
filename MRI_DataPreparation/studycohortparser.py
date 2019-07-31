import json
import os

Study_coordinator = []
path = 'MRI_DataPreparation/data/StudyCohort_prostate_mri.json'
with open(path, 'r') as f:
    for i in range(len(json.load(f))):
        with open(path, 'r') as f:
            jsondata = json.load(f)[i]
            path2 = 'MRI_DataPreparation/MRI_cases_test'
            tgzpath = os.path.join(path2, os.path.basename(jsondata["filename"]))
            print('Entering tgz directory: ' + tgzpath)
            if os.path.exists(tgzpath):
                Study_coordinator.append(jsondata.copy())
                print('Saving directory.')
with open('MRI_DataPreparation/data/StudyCohort_cut.json', 'w') as outfile:
    json.dump(Study_coordinator, outfile)
print('Script complete. Exiting program now.')