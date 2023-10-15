import os
import shutil
import os.path as osp
import cv2
from ploy import *


save_dir = '/home/cqjtu/Public/gligen_image'
file_name = ('/home/cqjtu/Documents/dataset/v1.0-mini/samples/CAM_BACK/n008-2018-08-28-16-43-51-0400__CAM_BACK__1535489306437558.jpg')
class_name = f"{file_name.split('__')[-1]}_bus"
save_dir = osp.join(save_dir, class_name)
dir_path = f'/home/cqjtu/Public/gligen_image/{class_name}'


def check_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


dirs = ['/home/cqjtu/Documents/dataset/v1.0-mini/samples',
        '/home/cqjtu/Documents/dataset/v1.0-mini/samples_5',
        '/home/cqjtu/Documents/dataset/v1.0-mini/samples_4',
        '/home/cqjtu/Documents/dataset/v1.0-mini/samples_3',
        ]
dir_png = ['/home/cqjtu/Documents/dataset/v1.0-mini/samples_2',
           '/home/cqjtu/Documents/dataset/v1.0-mini/samples_1', ]


check_dir(save_dir)

for dir_ in dirs:
    file = osp.join(dir_, file_name.split('/')[-2], file_name.split('/')[-1])
    try:
        save_path = osp.join(save_dir, f"{dir_.split('/')[-1]}.jpg")
        shutil.copyfile(file, save_path)
    except:
        file_name.replace('jpg', 'png')
        save_path = osp.join(save_dir, f"{dir_.split('/')[-1]}.jpg")
        shutil.copyfile(file, save_path)

for dir_ in dir_png:
    file = osp.join(dir_, file_name.split('/')[-2], file_name.split('/')[-1].replace('jpg', 'png'))
    save_path = osp.join(save_dir, f"{dir_.split('/')[-1]}.png")
    shutil.copyfile(file, save_path)
path = '/home/cqjtu/Documents/dataset/v1.0-mini/samples_'
file = osp.join(path, file_name.split('/')[-2], 'new', file_name.split('/')[-1])
shutil.copyfile(file, osp.join(save_dir, f"samples_.jpg"))

# ploy images
images = [cv2.imread(os.path.join(dir_path, file)) for file in os.listdir(dir_path)]
images = [resize_img(img, (800, 600)) for img in images]
poly(images, 7, 1, dir_path)
