import glob
import os
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import subprocess

#import TensorBox/evaluate
import classification
import crop_image



#Detection paths and filenames
DETECTION_MODEL_DIR = "trainedNetworks/TensorBoxNetworks/7500iter"
JSON_FILE_PATH = DETECTION_MODEL_DIR + "/save.ckpt-7500.val_boxes.json"
IMAGE_NAME = "elgesetergate.png"
#IMAGE_NAME = "00023.ppm"


SAVE_CROPPED_IMG_PATH = DETECTION_MODEL_DIR + "/cropped_images"
FILE_FORMAT = ".png" #The file format of the image(s) containing detected signs

DETECTION_MODEL = DETECTION_MODEL_DIR + "/save.ckpt-7500"
EMPTY_JSON_FILE = "datasets/detection/single_image/val_boxes.json"


#Classification paths and filenames
CLASSIFICATION_MODEL_DIR = "trainedNetworks/ClassificationNetworks/1001iter_72acc"
CLASSIFIED_IMAGES_SAVE_PATH = CLASSIFICATION_MODEL_DIR + "/classified_signs"


IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    #Sign detection
    print("SIGN DETECTION")
    bashCommand = "python TensorBox/detection.py --weights " + DETECTION_MODEL + " --image_dir " + EMPTY_JSON_FILE + " --image_name " + IMAGE_NAME
    #os.system(bashCommand)

    result = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    print("SIGN DETECTION DONE")

    print("CROPPING IMAGES")
    #Crop signs out of the images gotten from detection
    cropped_images = crop_image.main(JSON_FILE_PATH)

    # Check if folder exists. If not, create
    if not os.path.exists(SAVE_CROPPED_IMG_PATH):
        os.makedirs(SAVE_CROPPED_IMG_PATH)

    i = 0
    for image in cropped_images:
        image.save(SAVE_CROPPED_IMG_PATH + "/cropped_image_" + str(i) + FILE_FORMAT) #save image
        i += 1

    print("CROPPING DONE")
    #SIGN RECOGNITION
    #Load images of detected signs from the TensorBox network
    print("SIGN RECOGNITION")
    sign_images = load_data(SAVE_CROPPED_IMG_PATH)

    #Rescale
    sign_images_rescaled = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                     for image in sign_images]


    #Classify sign type
    input_image_dimension = [IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y]

    predicted_labels = classification.classify(sign_images_rescaled, CLASSIFICATION_MODEL_DIR, input_image_dimension)

    # EVALUATING THE TEST
    #TODO:

    #timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')

    if not os.path.exists(CLASSIFIED_IMAGES_SAVE_PATH):
        os.makedirs(CLASSIFIED_IMAGES_SAVE_PATH)

    i = 0

    for pl in predicted_labels:
        predicted_image = cropped_images[i]
        predicted_image.save(CLASSIFIED_IMAGES_SAVE_PATH + '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1

def load_data(data_dir):
    images = []
    for filename in glob.glob(data_dir+"/*"+ FILE_FORMAT):
        images.append(skimage.data.imread(filename)) #Loads the images as a list of numpy arrays

    return images

def save_numpy_array_as_image(array, save_dir, filename):
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(save_dir + filename)

main()