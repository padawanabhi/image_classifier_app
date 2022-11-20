import os

import cv2
import numpy as np


def create_dataset_from_images(images_folder) -> list:
    images = []
    classes = []

    for folder in os.listdir(images_folder):
        for file in os.listdir(images_folder + folder):
            image = cv2.imread(images_folder + folder + '/' + file, cv2.COLOR_RGB2BGR)
            image = np.array(image)
            image = image.astype('float32')
            image = image/255
            images.append(image)
            classes.append(folder)

    return images, classes


def preprocess_image_for_model(path) -> np.array:
    new_image = cv2.imread(path)
    new_image = new_image.astype('float32')
    new_image = new_image/255
    corrected_image = np.expand_dims(new_image, axis=0)
    return corrected_image



def create_target_encoding(target_list) -> dict:
    
    target_dict = {target: index for index, target in enumerate(np.unique(target_list))}
    print(target_dict)

    numeric_targets = [target_dict[target_class] for target_class in target_list]

    return numeric_targets



def decode_numeric_target(numeric_class) -> str:
    
    output_dict = {'cap': 0, 'phone': 1, 'shoe': 2, 'small_box': 3, 'stuffed_toy': 4}

    numeric_dict = {value: key for key, value in output_dict.items()}

    for key, value in numeric_dict.items():
        if key == numeric_class:
            return value


def decode_four_target(numeric_class) -> str:
    
    output_dict = {'phone': 0, 'shoe': 1, 'small_box': 2, 'stuffed_toy': 3}

    numeric_dict = {value: key for key, value in output_dict.items()}

    for key, value in numeric_dict.items():
        if key == numeric_class:
            return value


def decode_first_model(numeric_class) -> str:
    
    output_dict = {'coffee_mug': 0, 'phone': 1, 'small_box': 2}

    numeric_dict = {value: key for key, value in output_dict.items()}

    for key, value in numeric_dict.items():
        if key == numeric_class:
            return value

