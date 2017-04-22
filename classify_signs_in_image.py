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
import crop_image

ROOT_PATH = "output"
IMAGE_NAME = "00420.ppm"

#Detection paths and filenames
DETECTION_MODEL_DIR = "TensorBox/output/tensorbox_7500"
JSON_FILE_PATH = "TensorBox/output/tensorbox_7500/save.ckpt-7500.test_boxes.json"
SAVE_CROPPED_IMG_PATH = "TensorBox/output/tensorbox_7500/cropped_images"
FILE_FORMAT = ".ppm" #The file format of the image(s) containing detected signs

#Classification paths and filenames
CLASSIFICATION_MODEL_DIR = "BelgiumTS/2017_04_22_12.24_1001"

IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    #Sign detection
        #TODO create a json-file and an image with detected signs


    #Crop signs out of the images gotten from detection
    cropped_images = crop_image.main(JSON_FILE_PATH)

    # Check if folder exists. If not, create
    if not os.path.exists(SAVE_CROPPED_IMG_PATH):
        os.makedirs(SAVE_CROPPED_IMG_PATH)

    i = 0
    for image in cropped_images:
        image.save(SAVE_CROPPED_IMG_PATH + "/cropped_image_" + str(i) + FILE_FORMAT)
        i += 1


    #Sign recognition
    #Load images of detected signs from the TensorBox network
    sign_images = load_data(ROOT_PATH +"/"+SAVE_CROPPED_IMG_PATH)

    #Rescale
    sign_images_rescaled = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                     for image in sign_images]

    #Classify sign type
    input_image_dimension = [IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y]

    predicted_labels = classification.classify(sign_images_rescaled, CLASSIFICATION_MODEL_DIR, input_image_dimension)

    # EVALUATING THE TEST
    #TODO:

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