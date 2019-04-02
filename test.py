import os
import torch
import time
import numpy as np
import skimage.transform
from PIL import Image
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from main_monodepth_pytorch import Model

dict_parameters_test = edict({'data_dir': 'data/test',
                              'model_path': 'data/models/monodepth_resnet18_001.pth',
                              'output_directory': 'data/output',
                              'input_height': 256,
                              'input_width': 512,
                              'model': 'resnet18_md',
                              'pretrained': True,
                              'mode': 'test',
                              'device': 'cuda:0',
                              'input_channels': 3,
                              'num_workers': 4,
                              'use_multiple_gpu': False})
model_test = Model(dict_parameters_test)
a = time.time()
model_test.test()
b = time.time()
print("time = ", b - a)

disp = np.load('data/output/disparities.npy')

for i in range(disp.shape[0]):
    disp_to_img = skimage.transform.resize(disp[i].squeeze(), [413, 512], mode='constant')
    print("size = ", disp_to_img.shape)
    print("min = ", disp_to_img.min())
    print("max = ", disp_to_img.max())
    print("TYPE IMG = ", type(disp_to_img))

    # depth = (disp_to_img - disp_to_img.min()) / (disp_to_img.max() - disp_to_img.min())
    # depth = 1.0 - depth
    # depth = depth * 255.0
    # depth = depth.astype(np.uint8)

    # img = Image.fromarray(depth)
    # img.save(os.path.join(dict_parameters_test.output_directory, 'pred_'+str(i)+'.png'))

    plt.imsave(os.path.join(dict_parameters_test.output_directory,
                            'pred_' + str(i) + '.png'), disp_to_img, cmap='gray')