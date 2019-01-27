#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : my_model.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2019/1/19 15:57
@version: 1.0
@desc  : 
"""

from crnn_utils.draw_image import inference
import time
from crnn_utils.data_augmentation import CRNNAugmentor,AugmentationConfig
if __name__ == '__main__':
    # 添加augmentor配置
    augmentation_config = AugmentationConfig()
    # augmentation_config.rotate_image_params = {'max_angel': 0}  # no rotation!
    augmentation_config.add_table_lines_params = {'add_line_prob': 0.7, 'max_line_thickness': 4}
    augmentation_config.rotate_image_params = {'max_angel': 3}
    augmentation_config.affine_transform_shear_params = {'a': 3}
    augmentation_config.affine_transform_change_aspect_ratio_params = {'ratio': 0.9}
    augmentation_config.brighten_image_params = {'min_alpha': 0.6}
    augmentation_config.darken_image_params = {'min_alpha': 0.6}
    augmentation_config.add_color_filter_params = {'min_alpha': 0.6}
    augmentation_config.add_random_noise_params = {'min_masks': 70, 'max_masks': 90}
    augmentation_config.add_color_font_effect_params = {'beta': 0.6, 'max_num_lines': 90}
    augmentation_config.add_erode_edge_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 5}
    augmentation_config.add_resize_blur_effect_params = {'resize_ratio_range': (0.9, 1)}
    augmentation_config.add_gaussian_blur_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 5}
    augmentation_config.add_horizontal_motion_blur_effect_params = {'min_kernel_size': 5, 'max_kernel_size': 8}
    augmentation_config.add_vertical_motion_blur_effect_params = {'min_kernel_size': 5, 'max_kernel_size': 8}
    augmentation_config.add_random_circles_params = {'min_alpha': 0.6, 'max_num_circles': 25}
    augmentation_config.add_random_lines_params = {'min_alpha': 0.6, 'max_num_lines': 25}

    # augmentation (make the image look blurry)
    data_augmentor = CRNNAugmentor(augmentation_config)
    start = time.time()
    for i in range(128):
        image = inference("的什么玩意阿斯顿阿发斯蒂芬阿斯顿阿斯顿撒旦阿达水电费ad 发撒旦法爱迪生啊阿斯顿发送到是的", '/data/nfsdata/data/sunzijun/CV/more_fonts', data_augmentor)
    print(time.time() - start)
