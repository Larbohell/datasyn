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

ROOT_PATH = "output"
IMAGE = "datasets/detection/TestIJCNN2013/00103.ppm"
DETECTED_SIGNS_DIR = "FromTensorBox/detected_signs"
CLASSIFICATION_MODEL_DIR = "BelgiumTS/2017_04_22_12.24_1001"
DETECTION_MODEL = "trainedNetworks/TensorBoxNetworks/7500iter/save.ckpt-7500"
EMPTY_JSON_FILE = "datasets/detection/single_image/val_boxes.json"

IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    #Sign detection

    bashCommand = "python TensorBox/detection.py --weights " + DETECTION_MODEL + " --image_dir " + EMPTY_JSON_FILE
    #os.system(bashCommand)

    result = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
    result.stdout.decode('utf-8')

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