import json
import os
import random


def read_json(file):
    with open(file, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        return fcc_data


def save_json(file_name, data):
    with open(file_name, "w") as dump_f:
        json.dump(data, dump_f, indent=4)


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'pedestrian',
                  'motorcycle', 'traffic_cone', 'barrier')
color_categories = ('white', 'red', 'yellow', 'blue', 'green', 'Purple', 'black')
# file='/home/cqjtu/Documents/dataset/gligen/b_left/bl_images_cation.json'
other = ""
samples_num = 'samples_1'
directory = '/media/cqjtu/PortableSSD/dataSet/nusenes/nuscenes/sample_src'
for root, dirs, files in os.walk(directory):
    for file in files:
        if 'new_images_caption.json' in file:
            new_data = dict()
            json_file = os.path.join(root, file)
            data = read_json(json_file)

            for k, v in data.items():
                for object in nus_categories:
                    if object in v:
                        _v = random.sample(color_categories, 1)[0] + ' ' + object
                        v = v.replace(object, _v) + ',' + _v
                        # print(v)
                new_data[k] = v + other
            json_dir = os.path.dirname(json_file)
            print(f'json dir:{json_dir}')
            save_json(os.path.join(json_dir, f'{samples_num}_new_caption.json'), new_data)
