#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : draw_image.py
@author: zijun
@contact : stefan_sun_cn@hotmail.com
@date  : 2019/1/23 20:23
@version: 1.0
@desc  : 
"""
import glob
import random

from PIL import Image, ImageFont, ImageDraw


def random_color_pairs():
    """
    随机选取颜色
    """
    bright_color = random.randint(125, 255), random.randint(125, 255), random.randint(125, 255)
    dark_color = random.randint(0, 125), random.randint(0, 125), random.randint(0, 125)

    return bright_color, dark_color


def draw_text_on_image(content: str, font_dir, image_height=32, image_width=300):
    """
    根据给定的文字，随机选取字体大小、字体颜色、字体、位置、生成图片
    """
    if random.random() < 0.1:
        font_paths = glob.glob(font_dir + '/rare/*')
    else:
        font_paths = glob.glob(font_dir + '/common/*')

    font_path = random.choice(font_paths)
    font_size = random.randint(18, 30)

    while True:
        font = ImageFont.truetype(font_path, font_size)
        content_size = font.getsize(content)
        max_x0 = image_width - content_size[0]
        max_y0 = image_height - content_size[1]

        # check whether the text is small enough
        if max_x0 > 0 and max_y0 > 0:
            break

        # reduce font size if the text is out of boundaries
        font_size -= 1

    # randomly set x0, y0 according to max_x0, max_y0
    x0, y0 = random.randint(0, max_x0), random.randint(0, max_y0)

    if random.random() < 0.1:
        text_color, background_color = random_color_pairs()
    else:
        background_color, text_color = random_color_pairs()

    image = Image.new(mode='RGB', size=(image_width, image_height), color=background_color)
    draw = ImageDraw.Draw(image)
    draw.text(xy=[x0, y0], text=content, font=font, fill=text_color)

    return image


def inference(input_text: str, font_dir: str, data_augmentor):
    """
    draw text on image, and inference the image with CRNN model
    :param input_text: input text string
    :param augmentation: whether to make the image look blurry
    :return: CRNN prediction (string)
    """
    # 画图
    image = draw_text_on_image(input_text,
                               font_dir,
                               image_height=32,
                               image_width=300)

    image = data_augmentor.augment_image(image)

    return image
