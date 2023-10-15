import os.path

import cv2
import numpy as np

# root='/home/cqjtu/Documents/dataset/gligen/'
# segpath=os.path.join(root,'dep')
# files=os.listdir(segpath)
# for file in files:
#     img=cv2.imread(os.path.join(segpath,file))
#     if np.ptp(img)>1:
#         print()
#         print(f'max:{np.amax(img,(0,1,2))}  min:{np.min(img,(0,1,2))}')
#         print(f'ptp:{np.ptp(img)}')
#         print(f'median:{np.median(img)}')

import json
filedir='/media/cqjtu/PortableSSD/dataSet/nusenes/nuscenes/sample_src'

for dir in os.listdir(filedir):
    if "CAM" in dir:
        cam_dir=os.path.join(filedir,dir)
        savedict= {}
        with open(os.path.join(cam_dir,'images_cation.json'),'r') as f:
            data = json.load(f)

        for file in os.listdir(cam_dir):
            if 'img' in file:
                # for fi in os.listdir(os.path.join(cam_dir,file)):
                #     savedict[fi]=data[fi]
                savedict = {key: data[key] for key in os.listdir(os.path.join(cam_dir,file)) if key in data}

        with open(os.path.join(cam_dir,'new_images_caption.json'),'w') as ff:
            json.dump(savedict,ff)
