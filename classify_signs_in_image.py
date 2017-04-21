import glob
import os
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime

#import TensorBox/evaluate
import classification

ROOT_PATH = "output"
IMAGE = "datasets/detection/TestIJCNN2013/00103.ppm"
DETECTED_SIGNS_DIR = "FromTensorBox/detected_signs"
CLASSIFICATION_MODEL_DIR = "BelgiumTS/2017_04_21_21.14_101"

IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    #Sign detection
        #DO STUFF, save detected signs into DETECTED_SIGNS_DIR

    #Sign recognition
    #Load images of detected signs from the TensorBox network
    sign_images = load_data(ROOT_PATH +"/"+DETECTED_SIGNS_DIR)

    #Rescale
    sign_images_rescaled = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                     for image in sign_images]

    #Classify sign type
    input_image_dimension = [IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y]

    predicted_labels = classification.classify(sign_images_rescaled, CLASSIFICATION_MODEL_DIR, input_image_dimension)

    # EVALUATING THE TEST
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')
    save_dir = 'output/' + DETECTED_SIGNS_DIR + '/predictions_' + timestamp

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0

    for pl in predicted_labels:
        predicted_image = sign_images_rescaled[i]
        save_numpy_array_as_image(predicted_image, save_dir, '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1

def load_data(data_dir):
    images = []
    for filename in glob.glob(data_dir+"/*.ppm"):
        images.append(skimage.data.imread(filename)) #Loads the images as a list of numpy arrays

    return images

def save_numpy_array_as_image(array, save_dir, filename):
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(save_dir + filename)

main()