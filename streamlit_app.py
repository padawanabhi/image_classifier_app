import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
from keras.models import load_model
from load_images import decode_numeric_target, decode_four_target, decode_first_model

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224




def predict_class(image):
    image_mat = cv2.resize(image, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = np.array(image_mat)
    image_array = image_array.astype('float32')
    image_array = image_array/255
    new_array = np.expand_dims(image_array, axis=0)
    print(new_array.shape)
    model = load_model('./pretrained_models/first_model.h5')
    pred = model.predict(new_array)
    print(pred)
    print(np.argmax(pred))
    prediction = decode_first_model(np.argmax(pred))

    return prediction

######################################################

def predict_firstnew(image):
    image_mat = cv2.resize(image, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = np.array(image_mat)
    image_array = image_array.astype('float32')
    image_array = image_array/255
    new_array = np.expand_dims(image_array, axis=0)
    model = load_model('./pretrained_models/first_new_model.h5')
    pred = model.predict(new_array)
    print(pred)
    print(np.argmax(pred))
    prediction = decode_first_model(np.argmax(pred))

    return prediction

####################################################

def predict_augmented(image):
    image_mat = cv2.resize(image, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = np.array(image_mat)
    image_array = image_array.astype('float32')
    image_array = image_array/255
    new_array = np.expand_dims(image_array, axis=0)
    model = load_model('./pretrained_models/augmented_model.h5')
    pred = model.predict(new_array)
    print(pred)
    print(np.argmax(pred))
    prediction = decode_first_model(np.argmax(pred))

    return prediction

##############################################

def predict_fourclass(image):
    image_mat = cv2.resize(image, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = np.array(image_mat)
    image_array = image_array.astype('float32')
    image_array = image_array/255
    new_array = np.expand_dims(image_array, axis=0)
    model = load_model('./pretrained_models/fourclass_model.h5')
    pred = model.predict(new_array)
    print(pred)
    print(np.argmax(pred))
    prediction = decode_four_target(np.argmax(pred))

    return prediction    

########################################################

def predict_new(image):
    image_mat = cv2.resize(image, dsize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_array = np.array(image_mat)
    image_array = image_array.astype('float32')
    image_array = image_array/255
    new_array = np.expand_dims(image_array, axis=0)
    model = load_model('./pretrained_models/new_model.h5')
    pred = model.predict(new_array)
    print(pred)
    print(np.argmax(pred))
    prediction = decode_numeric_target(np.argmax(pred))

    return prediction 

####################################################################


if __name__ == '__main__':

    st.title('Image Classifier App')

    with st.sidebar:
        model_choice = st.radio(
                        "Choose an model:",
                        ('First','Augmented', 'Fourclass', 'New', 'First_new'))
        choice = st.radio(
                        "Choose an option:",
                        ('None','Take a picture', 'Upload a picture'))

    if choice == 'Take a picture':
        st.write('Great! Lets take a picture')

        picture = st.camera_input(label="Take an image to classify")
        if picture is not None:
            img_load = Image.open(picture)
            img_load.save('temp.png')
            img = cv2.imread('temp.png', cv2.COLOR_BGR2RGB)
            if model_choice == 'First':
                prediction = predict_class(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'Fourclass':
                prediction = predict_fourclass(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'New':
                prediction = predict_new(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'First_new':
                prediction = predict_firstnew(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'Augmented':
                prediction = predict_augmented(np.array(img))
                st.write(f'The object is a {prediction}')     


    elif choice == 'Upload a picture':
        st.write('Awesome! Let upload the picture then')
        uploaded_file = st.file_uploader("Choose a image to classify")
        if uploaded_file is not None:
            st.image(uploaded_file)
            img_load = Image.open(uploaded_file)
            img_load.save('temp.png')
            img = cv2.imread('temp.png', cv2.COLOR_RGB2BGR)
            if model_choice == 'First':
                prediction = predict_class(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'Fourclass':
                prediction = predict_fourclass(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'New':
                prediction = predict_new(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'First_new':
                prediction = predict_firstnew(np.array(img))
                st.write(f'The object is a {prediction}')
            elif model_choice == 'Augmented':
                prediction = predict_augmented(np.array(img))
                st.write(f'The object is a {prediction}')   

    
    else:
        st.write("You haven't uploaded any image file or taken a picture to classify")

