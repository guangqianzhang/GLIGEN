import os
import cv2
import numpy as np


def resize_img(img, size):
    img = cv2.resize(img, size, interpolation=3 )
    return img

def save_image(output_path,image):
    root='/home/cqjtu/GLIGEN/scrpits/proceesimg'
    path=os.path.join(root,output_path)
    path=os.path.dirname(path)
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(output_path, image)
def half_spilt(imag_path, path):

    image=cv2.imread(imag_path)
    # 获取图片的尺寸
    height, width, _ = image.shape
    # 计算分割点
    split_point = width // 2

    image_1 = image[:, :split_point]
    image_2 = image[:, split_point:]

    # 保存分割后的图像
    save_image(os.path.join(path,'image1.jpg'),image_1)
    save_image(os.path.join(path,'image2.jpg'),image_2)


def poly(images, num_rows, num_cols,path, spacing=10):
    # 创建一个大画布来容纳所有图片
    canvas_height = num_rows * (images[0].shape[0] + spacing) - spacing
    canvas_width = num_cols * (images[0].shape[1] + spacing) - spacing
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 依次加载图片并按照布局放置
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        # image = cv2.imread(image)

        y = row * (image.shape[0] + spacing)
        x = col * (image.shape[1] + spacing)

        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image

    cv2.imwrite(path + '/save.jpg', canvas)


# path='/media/cqjtu/PortableSSD/dataSet/GLIGEN/2023810/test/tag16/00370001.png'
# half_spilt(path,'proceedimg')


path='/home/cqjtu/Public/weather/sunny'

images=[cv2.imread(os.path.join(path,file)) for file in os.listdir(path)]
images=[resize_img(img,(800,600)) for  img in images]
poly(images,len(images),1,path)


