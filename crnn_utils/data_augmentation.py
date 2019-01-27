# encoding: utf-8
"""
@author: Junxu Lu
@contact: junxu_lu@shannonai.com

@version: 1.0
@file: data_augmentation.py
@time: 11/14/18 18:32

CRNN, CTPN 数据增强算法
"""

import random

import cv2
import mahotas
import numpy as np
from PIL import Image


class AugmentationConfig(object):
    def __init__(self):
        # default: level 2 (very intense augmentation)
        self.add_table_lines_params = {'add_line_prob': 0.5, 'max_line_thickness': 3}
        self.add_underline_params = {'max_line_thickness': 3}
        self.rotate_image_params = {'max_angel': 3}
        self.affine_transform_shear_params = {'a': 1.5}
        self.affine_transform_change_aspect_ratio_params = {'ratio': 0.7}
        self.brighten_image_params = {'min_alpha': 0.6}
        self.darken_image_params = {'min_alpha': 0.6}
        self.add_color_filter_params = {'min_alpha': 0.6}
        self.add_random_noise_params = {'min_masks': 40, 'max_masks': 100}
        self.add_color_font_effect_params = {'beta': 0.6, 'max_num_lines': 50}
        self.add_erode_edge_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 3}
        self.add_resize_blur_effect_params = {'resize_ratio_range': (0.5, 0.7)}
        self.add_gaussian_blur_effect_params = {'kernel_size': (3, 3), 'max_sigmaX': 3}
        self.add_horizontal_motion_blur_effect_params = {'min_kernel_size': 1, 'max_kernel_size': 2}
        self.add_vertical_motion_blur_effect_params = {'min_kernel_size': 1, 'max_kernel_size': 2}
        self.add_random_circles_params = {'min_alpha': 0.6, 'max_num_circles': 20}
        self.add_random_lines_params = {'min_alpha': 0.6, 'max_num_lines': 20}


class AugmentationFunctions(object):
    @staticmethod
    def random_color():
        return random.randint(0, 255), random.randint(10, 255), random.randint(10, 255)

    @staticmethod
    def get_background_color(image_array: np.ndarray):
        return int(image_array[0, 0, 0]), int(image_array[0, 0, 1]), int(image_array[0, 0, 2])

    @staticmethod
    def add_random_padding(image: Image.Image, output_size=(300, 32), scale_ratio_range=(0.6, 0.99),
                           method=Image.ANTIALIAS) -> Image.Image:
        # 随机边缘填充（缩放文字）
        color_mode = image.mode
        assert color_mode == 'RGB' or 'L'
        scale_ratio = random.uniform(*scale_ratio_range)
        im_aspect = float(image.size[0]) / float(image.size[1])
        out_aspect = float(output_size[0]) / float(output_size[1])
        if im_aspect >= out_aspect:
            scaled = image.resize((int(scale_ratio * output_size[0]),
                                   int(scale_ratio * (float(output_size[0]) / im_aspect) + 0.5)), method)
        else:
            scaled = image.resize((int(scale_ratio * (float(output_size[1]) * im_aspect) + 0.5),
                                   int(scale_ratio * output_size[1])), method)
        offset_w = int(random.uniform(0, output_size[0] - scaled.size[0]))
        offset_h = int(random.uniform(0, output_size[1] - scaled.size[1]))
        offset = (offset_w, offset_h)
        if color_mode == 'L':
            back = Image.new(image.mode, output_size, (255,))
        else:
            back = Image.new(image.mode, output_size, (255, 255, 255))
        back.paste(scaled, offset)

        return back

    @staticmethod
    def add_table_lines(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 随机加 0-4 条表格线
        add_line_prob = augmentation_config.add_table_lines_params['add_line_prob']
        max_line_thickness = augmentation_config.add_table_lines_params['max_line_thickness']

        h, w = image_array.shape[:2]
        line_thickness = random.randint(1, max_line_thickness)
        line_color = random.randint(0, 10)
        # 上表格线
        if random.random() < add_line_prob:
            line_y = random.randint(0, round(h / 3))
            cv2.line(img=image_array, pt1=(0, line_y), pt2=(w, line_y),
                     color=line_color, thickness=line_thickness)
        # 下表格线
        if random.random() < add_line_prob:
            line_y = random.randint(round(2 * h / 3), h)
            cv2.line(img=image_array, pt1=(0, line_y), pt2=(w, line_y),
                     color=line_color, thickness=line_thickness)
        # 左表格线
        if random.random() < add_line_prob:
            line_x = random.randint(0, round(w / 6))
            cv2.line(img=image_array, pt1=(line_x, 0), pt2=(line_x, h),
                     color=line_color, thickness=line_thickness)
        # 右表格线
        if random.random() < add_line_prob:
            line_x = random.randint(round(5 * w / 6), w)
            cv2.line(img=image_array, pt1=(line_x, 0), pt2=(line_x, h),
                     color=line_color, thickness=line_thickness)

        return image_array

    @staticmethod
    def add_underline(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 加下划线
        max_line_thickness = augmentation_config.add_underline_params['max_line_thickness']

        h, w = image_array.shape[:2]
        line_thickness = random.randint(1, max_line_thickness)
        line_color = random.randint(0, 10)
        line_y = random.randint(round(4 * h / 5), h)

        cv2.line(img=image_array,
                 pt1=(random.randint(0, w), line_y),
                 pt2=(random.randint(0, w), line_y),
                 color=line_color, thickness=line_thickness)

        return image_array

    @staticmethod
    def rotate_image(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 旋转图片
        max_angel = augmentation_config.rotate_image_params['max_angel']

        if max_angel > 2:
            max_resize_scale = 1
        else:
            max_resize_scale = 1.02
        rotation_angle = random.uniform(-max_angel, max_angel)
        image_h, image_w = image_array.shape[0], image_array.shape[1]
        rotation_center = (image_w // 2, image_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center=rotation_center, angle=-rotation_angle,
                                                  scale=random.uniform(0.85, max_resize_scale))
        rotated_array = cv2.warpAffine(image_array, rotation_matrix, image_array.shape[1::-1],
                                       flags=cv2.INTER_LINEAR,
                                       borderValue=AugmentationFunctions.get_background_color(image_array))

        return rotated_array

    @staticmethod
    def affine_transform_shear(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # Affine Transform: Shear（斜体字效果）
        a = augmentation_config.affine_transform_shear_params['a']

        height, width, _ = image_array.shape
        shift = random.randint(0, int(a * height))

        origin_points = np.float32([[0, 0], [width, 0], [width, height]])
        new_points = np.float32([[shift, 0], [width, 0], [width - shift, height]])

        transform_matrix = cv2.getAffineTransform(origin_points, new_points)
        transformed_image_array = cv2.warpAffine(src=image_array,
                                                 M=transform_matrix,
                                                 dsize=(width, height),
                                                 borderValue=AugmentationFunctions.get_background_color(image_array))

        return transformed_image_array

    @staticmethod
    def affine_transform_change_aspect_ratio(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # Affine Transform: Change Aspect Ratio（文字挤压拉伸效果）
        ratio = augmentation_config.affine_transform_change_aspect_ratio_params['ratio']

        height, width, _ = image_array.shape
        origin_points = np.float32([[0, 0], [width, 0], [0, height]])

        if random.random() < 0.5:
            new_points = np.float32([[0, 0], [width * random.uniform(ratio, 1), 0], [0, height]])
        else:
            new_points = np.float32([[0, 0], [width, 0], [0, height * random.uniform(ratio, 1)]])

        transform_matrix = cv2.getAffineTransform(origin_points, new_points)
        transformed_image_array = cv2.warpAffine(src=image_array,
                                                 M=transform_matrix,
                                                 dsize=(width, height),
                                                 borderValue=AugmentationFunctions.get_background_color(image_array))

        return transformed_image_array

    @staticmethod
    def brighten_image(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 滤镜：白色滤镜（提高亮度）
        min_alpha = augmentation_config.brighten_image_params['min_alpha']

        alpha = random.uniform(min_alpha, 1.0)
        beta = 1 - alpha
        gamma = 0
        h, w = image_array.shape[:2]
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        cv2.rectangle(img=bg_mark, pt1=(0, 0), pt2=(w, h),
                      color=(255, 255, 255), thickness=-1)
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array

    @staticmethod
    def darken_image(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 滤镜：黑色滤镜（降低亮度）
        min_alpha = augmentation_config.darken_image_params['min_alpha']

        alpha = random.uniform(min_alpha, 1.0)
        beta = 1 - alpha
        gamma = 0
        h, w = image_array.shape[:2]
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        cv2.rectangle(img=bg_mark, pt1=(0, 0), pt2=(w, h),
                      color=(0, 0, 0), thickness=-1)
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array

    @staticmethod
    def add_color_filter(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 滤镜：彩色滤镜
        min_alpha = augmentation_config.add_color_filter_params['min_alpha']

        alpha = random.uniform(min_alpha, 1.0)
        beta = 1 - alpha
        gamma = 0
        h, w = image_array.shape[:2]
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        cv2.rectangle(img=bg_mark, pt1=(0, 0), pt2=(w, h),
                      color=AugmentationFunctions.random_color(), thickness=-1)
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array

    @staticmethod
    def change_color(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真：背景变色
        if len(image_array.shape) == 3:
            trans = np.array([random.random() for _ in range(9)]).reshape(3, 3)
            image_array = np.dot(image_array, trans).clip(max=255)
        elif len(image_array.shape) == 2:
            trans = np.array(random.uniform(0.1, 1))
            image_array = np.dot(image_array, trans).clip(max=255)
            # randomly invert
            if random.random() > 0.5:
                image_array = np.abs(225 - image_array)
        else:
            raise ValueError('wrong color mode, use RGB or L')
        image_array = image_array.astype('uint8')

        return image_array

    @staticmethod
    def add_random_noise(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真：随机噪声
        min_masks = augmentation_config.add_random_noise_params['min_masks']
        max_masks = augmentation_config.add_random_noise_params['max_masks']

        n_masks = random.randint(min_masks, max_masks)
        random_mask = np.random.rand(*image_array.shape) * n_masks
        random_mask[image_array < 50] = 0
        random_mask = random_mask.astype('uint8')
        noise_image_array = image_array - random_mask
        noise_image_array[noise_image_array < 0] = 0

        return noise_image_array

    @staticmethod
    def add_color_font_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真：字体变色(彩色字体/部分缺失效果)
        beta = augmentation_config.add_color_font_effect_params['beta']
        max_num_lines = augmentation_config.add_color_font_effect_params['max_num_lines']

        alpha = 1
        gamma = 0
        h, w = image_array.shape[:2]
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        num_lines = random.randint(1, max_num_lines)

        for i in range(num_lines):
            x0 = random.randint(0, w)
            y0 = random.randint(0, h)
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            cv2.rectangle(img=bg_mark, pt1=(x0, y0), pt2=(x1, y1),
                          color=AugmentationFunctions.random_color(), thickness=-1)
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array

    @staticmethod
    def add_erode_edge_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真：腐蚀边缘(模糊+二值化)
        kernel_size = augmentation_config.add_erode_edge_effect_params['kernel_size']
        max_sigmaX = augmentation_config.add_erode_edge_effect_params['max_sigmaX']

        image_array = cv2.GaussianBlur(src=image_array, ksize=kernel_size, sigmaX=np.random.randint(1, max_sigmaX))
        T = mahotas.thresholding.otsu(image_array)
        thresh = image_array.copy()
        thresh[thresh > T] = 255
        thresh[thresh < 255] = 0
        image_array = thresh

        return image_array

    @staticmethod
    def add_resize_blur_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真-模糊：缩放模糊
        resize_ratio_range = augmentation_config.add_resize_blur_effect_params['resize_ratio_range']

        h, w = image_array.shape[:2]
        resize_ratio = random.uniform(*resize_ratio_range)
        image_array = cv2.resize(image_array, (int(w * resize_ratio), int(h * resize_ratio)))
        image_array = cv2.resize(image_array, (w, h))

        return image_array

    @staticmethod
    def add_gaussian_blur_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真-模糊：高斯模糊
        kernel_size = augmentation_config.add_gaussian_blur_effect_params['kernel_size']
        max_sigmaX = augmentation_config.add_gaussian_blur_effect_params['max_sigmaX']

        image_array = cv2.GaussianBlur(src=image_array, ksize=kernel_size, sigmaX=np.random.randint(1, max_sigmaX))

        return image_array

    @staticmethod
    def add_horizontal_motion_blur_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真-模糊：水平运动模糊
        min_kernel_size = augmentation_config.add_horizontal_motion_blur_effect_params['min_kernel_size']
        max_kernel_size = augmentation_config.add_horizontal_motion_blur_effect_params['max_kernel_size']

        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image_array = cv2.filter2D(image_array, -1, kernel_motion_blur)

        return image_array

    @staticmethod
    def add_vertical_motion_blur_effect(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 失真-模糊：垂直运动模糊
        min_kernel_size = augmentation_config.add_vertical_motion_blur_effect_params['min_kernel_size']
        max_kernel_size = augmentation_config.add_vertical_motion_blur_effect_params['max_kernel_size']

        kernel_size = random.randint(min_kernel_size, max_kernel_size)
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        image_array = cv2.filter2D(image_array, -1, kernel_motion_blur)

        return image_array

    @staticmethod
    def add_random_circles(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 背景：随机添加圆圈
        min_alpha = augmentation_config.add_random_circles_params['min_alpha']
        max_num_circles = augmentation_config.add_random_circles_params['max_num_circles']

        alpha = random.uniform(min_alpha, 1.0)
        beta = 1 - alpha
        gamma = 0
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        num_circles = random.randint(1, max_num_circles)
        for i in range(num_circles):
            radius = np.random.randint(5, high=200)
            pt = np.random.randint(0, high=300, size=(2,))
            cv2.circle(img=bg_mark, center=tuple(pt), radius=radius, color=AugmentationFunctions.random_color(),
                       thickness=-1)
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array

    @staticmethod
    def add_random_lines(image_array: np.ndarray, augmentation_config) -> np.ndarray:
        # 背景：随机添加椭圆曲线
        min_alpha = augmentation_config.add_random_lines_params['min_alpha']
        max_num_lines = augmentation_config.add_random_lines_params['max_num_lines']

        alpha = random.uniform(min_alpha, 1.0)
        beta = 1 - alpha
        gamma = 0
        h, w = image_array.shape[:2]
        bg_mark = np.zeros(image_array.shape, np.uint8) + 255
        num_lines = random.randint(1, max_num_lines)
        for i in range(num_lines):
            x0 = random.randint(0, w)
            y0 = random.randint(0, h)
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            start_angle = random.randint(0, 90)
            end_angle = random.randint(0, 90) + start_angle
            cv2.ellipse(img=bg_mark, center=(x0, y0), axes=(x1, y1), angle=random.randint(0, 180),
                        startAngle=start_angle, endAngle=end_angle,
                        color=AugmentationFunctions.random_color(), thickness=random.randint(1, 10))
        image_array = cv2.addWeighted(image_array, alpha, bg_mark, beta, gamma)

        return image_array


class CRNNAugmentor(object):
    def __init__(self, augmentation_config):
        self.augmentation_config = augmentation_config

        self.use_affine_functions = True
        self.use_add_underline = True
        self.use_add_table_lines = True
        self.use_rotate_image = True
        self.use_reduce_fidelity_functions = True
        self.use_add_random_circles = True
        self.use_add_random_lines = True

        self.use_affine_functions_prob = 0.1
        self.use_add_underline_prob = 0.1
        self.use_add_table_lines_prob = 0.1
        self.use_rotate_image_prob = 0.5
        self.use_reduce_fidelity_functions_prob = 0.5
        self.use_add_random_circles_prob = 0.1
        self.use_add_random_lines_prob = 0.1

        # 仿射变换函数
        self.affine_functions = [
            AugmentationFunctions.affine_transform_shear,  # 斜体字效果
            AugmentationFunctions.affine_transform_change_aspect_ratio,  # 文字挤压拉伸效果
        ]

        # 模糊函数
        self.blur_functions = [
            AugmentationFunctions.add_resize_blur_effect,  # 缩放模糊
            AugmentationFunctions.add_gaussian_blur_effect,  # 高斯模糊
            AugmentationFunctions.add_vertical_motion_blur_effect,  # 垂直运动模糊
            AugmentationFunctions.add_horizontal_motion_blur_effect,  # 水平运动模糊
        ]

        # 失真函数
        self.reduce_fidelity_functions = [
            AugmentationFunctions.change_color,  # 背景变色
            AugmentationFunctions.add_random_noise,  # 随机噪声
            AugmentationFunctions.add_color_font_effect,  # 字体变色
            AugmentationFunctions.add_erode_edge_effect,  # 腐蚀边缘
            AugmentationFunctions.brighten_image,  # 滤镜：提高亮度
            AugmentationFunctions.darken_image,  # 滤镜：降低亮度
            AugmentationFunctions.add_color_filter,  # 滤镜：彩色滤镜
            random.choice(self.blur_functions),  # 模糊
        ]

    def augment_image(self, image: Image.Image):
        image_array = np.array(image)

        # 仿射变换（斜体字/文字挤压拉伸）
        if self.use_affine_functions and random.random() < self.use_affine_functions_prob:
            image_array = random.choice(self.affine_functions)(image_array, self.augmentation_config)

        # 下划线
        if self.use_add_underline and random.random() < self.use_add_underline_prob:
            image_array = AugmentationFunctions.add_underline(image_array, self.augmentation_config)

        # 表格线
        if self.use_add_table_lines and random.random() < self.use_add_table_lines_prob:
            image_array = AugmentationFunctions.add_table_lines(image_array, self.augmentation_config)

        # 旋转
        if self.use_rotate_image and random.random() < self.use_rotate_image_prob:
            image_array = AugmentationFunctions.rotate_image(image_array, self.augmentation_config)

        # 失真效果
        if self.use_reduce_fidelity_functions and random.random() < self.use_reduce_fidelity_functions_prob:
            image_array = random.choice(self.reduce_fidelity_functions)(image_array, self.augmentation_config)

        # 背景圆圈
        if self.use_add_random_circles and random.random() < self.use_add_random_circles_prob:
            image_array = AugmentationFunctions.add_random_circles(image_array, self.augmentation_config)

        # 背景线条
        if self.use_add_random_lines and random.random() < self.use_add_random_lines_prob:
            image_array = AugmentationFunctions.add_random_lines(image_array, self.augmentation_config)

        image = Image.fromarray(image_array)

        return image


class CTPNAugmentor(object):
    def __init__(self, augmentation_config):
        self.augmentation_config = augmentation_config

        # 模糊函数
        self.blur_functions = [
            AugmentationFunctions.add_resize_blur_effect,  # 缩放模糊
            AugmentationFunctions.add_gaussian_blur_effect,  # 高斯模糊
            AugmentationFunctions.add_vertical_motion_blur_effect,  # 垂直运动模糊
            AugmentationFunctions.add_horizontal_motion_blur_effect,  # 水平运动模糊
        ]

        # 失真函数
        self.reduce_fidelity_functions = [
            AugmentationFunctions.add_random_noise,  # 随机噪声
            AugmentationFunctions.add_erode_edge_effect,  # 腐蚀边缘
            AugmentationFunctions.brighten_image,  # 滤镜：提高亮度
            AugmentationFunctions.darken_image,  # 滤镜：降低亮度
            random.choice(self.blur_functions),  # 模糊
        ]

    def augment_image(self, image: Image.Image):
        image_array = np.array(image)

        # 失真效果
        if random.random() < 0.8:
            image_array = random.choice(self.reduce_fidelity_functions)(image_array, self.augmentation_config)
        # 背景圆圈
        if random.random() < 0.1:
            image_array = AugmentationFunctions.add_random_circles(image_array, self.augmentation_config)
        # 背景线条
        if random.random() < 0.1:
            image_array = AugmentationFunctions.add_random_lines(image_array, self.augmentation_config)

        image = Image.fromarray(image_array)

        return image
