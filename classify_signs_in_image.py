import glob
import os
import skimage.data
import skimage.transform
import skimage.exposure as exposure
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import subprocess
import shutil
import time

#import TensorBox/evaluate
import classification
import crop_image

TEST_DATA_DIR = "datasets/detection/single_image"
LABEL_TYPE = "GTSRB"
#LABEL_TYPE = "Belgium_TS"
#LABEL_TYPE = "" # Prints numerical lable instead of text label

#Detection paths and filenames
DETECTION_MODEL_DIR = "trainedNetworks/TensorBoxNetworks/7500iter"
JSON_FILE_PATH = DETECTION_MODEL_DIR + "/save.ckpt-7500.val_boxes.json"
#IMAGE_NAME = "elgesetergate.png"
#IMAGE_NAME = "00023.ppm"

SAVE_CROPPED_IMG_PATH = DETECTION_MODEL_DIR + "/cropped_images"
FILE_FORMAT = ".ppm" #The file format of the image(s) containing detected signs

DETECTION_MODEL = DETECTION_MODEL_DIR + "/save.ckpt-7500"
#EMPTY_JSON_FILE = "datasets/detection/single_image/val_boxes.json"
EMPTY_JSON_FILE = "datasets/detection/single_image/val_boxes.json"


#Classification paths and filenames
#CLASSIFICATION_MODEL_DIR = "trainedNetworks/ClassificationNetworks/1001iter_72acc"
CLASSIFICATION_MODEL_DIR = "trainedNetworks\ClassificationNetworks\1001iter_72acc"
CLASSIFIED_IMAGES_SAVE_PATH = CLASSIFICATION_MODEL_DIR + "/classified_signs"


IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    #image_name = "00023.ppm"
    #image_name = "elgesetergate.png"

    i = 0
    for filename in glob.glob(TEST_DATA_DIR + "/*" + FILE_FORMAT):
        image_name = os.path.basename(filename)
        detect_and_classify(image_name, i)
        print("Detect and recognize iter = ", i)
        i += 1

def pre_process_single_img(img):
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    img_y = (img_y/255).astype(np.float32)
    img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
    return img_y

def add_dimension(img):
    return img.reshape(img.shape + (1,))

def detect_and_classify(image_name, iter):
    # Sign detection
    print("SIGN DETECTION")
    bashCommand = "python TensorBox/detection.py --weights " + DETECTION_MODEL + " --image_dir " + EMPTY_JSON_FILE + " --image_name " + image_name
    # os.system(bashCommand)

    result = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    print("SIGN DETECTION DONE\n")

    print("CROPPING IMAGES")
    # Crop signs out of the images gotten from detection
    start_time = time.time()
    cropped_images = crop_image.main(JSON_FILE_PATH)

    # Check if folder exists. If not, create
    if not os.path.exists(SAVE_CROPPED_IMG_PATH):
        os.makedirs(SAVE_CROPPED_IMG_PATH)

    i = 0
    for image in cropped_images:
        image.save(SAVE_CROPPED_IMG_PATH + "/cropped_image_" + str(i) + "_" + FILE_FORMAT)  # save image
        i += 1

    print("CROPPING DONE\n")
    # SIGN RECOGNITION
    # Load images of detected signs from the TensorBox network
    print("SIGN RECOGNITION")
    sign_images = load_data(SAVE_CROPPED_IMG_PATH)

    # Rescale
    sign_images_rescaled = [skimage.transform.resize(pre_process_single_img(image), (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                            for image in sign_images]

    sign_images_rescaled = [add_dimension(image) for image in sign_images_rescaled]

    # Classify sign type
    input_image_dimension = [IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y]
    end_time = time.time()

    with open("timing_processing.txt", "a") as timerfile:
        processing_time = end_time - start_time
        timerfile.write(processing_time)
        timerfile.write("\n")

    predicted_labels = classification.classify(sign_images_rescaled, CLASSIFICATION_MODEL_DIR, input_image_dimension)

    # EVALUATING THE TEST

    # timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')

    if not os.path.exists(CLASSIFIED_IMAGES_SAVE_PATH):
        os.makedirs(CLASSIFIED_IMAGES_SAVE_PATH)

    i = 0

    for pl in predicted_labels:
        predicted_image = sign_images_rescaled[i]
        predicted_image.shape = (32,32);
        if LABEL_TYPE == "BelgiumTS":
            sign_type = label_to_type_BelgiumTS[pl]
        elif LABEL_TYPE == "GTSRB":
            sign_type = label_to_type_GTSRB[pl]
        else:
            sign_type = str(pl)
        #save_numpy_array_as_image(predicted_image,CLASSIFIED_IMAGES_SAVE_PATH,'/label_' + str(pl) + '_' + str(i) + "_" + str(iter) + '.png')
        save_numpy_array_as_image(predicted_image, CLASSIFIED_IMAGES_SAVE_PATH, '/label_' + sign_type + '_' + str(i) + "_" + str(iter) + '.png')

        #predicted_image.save(CLASSIFIED_IMAGES_SAVE_PATH + '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1

    print("SIGN RECOGNITION DONE")

    print("Preparing for next run and deleting cropped_images folder...")
    shutil.rmtree(SAVE_CROPPED_IMG_PATH)
    print(SAVE_CROPPED_IMG_PATH + " deleted")

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

label_to_type_BelgiumTS = {
    0: "fare_dump",
    1: "fare_hump",
    2: "fare_glatt",
    3: "fare_venstresving",
    4: "fare_høyresving",
    5: "fare_svingete_venstre",
    6: "fare_svingete_høyre",
    7: "fare_barn_og_mor",
    8: "fare_syklende",
    9: "fare_dyr",
    10: "fare_veiarbeid",
    11: "fare_traffiklys",
    12: "fare_grind",
    13: "fare_utropstegn",
    14: "fare_smal_veg",
    15: "fare_smal_venstre",
    16: "fare_smal_høyre",
    17: "fare_forkjørsveg",
    18: "fare_kryss",
    19: "fare_vikeplikt",
    20: "fare_vikteplikt_for_motgående",
    21: "stopp",
    22: "forbud_innkjøring_forbudt",
    23: "forbud_sykkel",
    24: "forbud_vekt",
    25: "forbud_lastebil",
    26: "forbud_bredde",
    27: "forbud_lengde",
    28: "forbud_blank",
    29: "forbud_venstresving",
    30: "forbud_høyresving",
    31: "forbud_forbikjøring",
    32: "forbud_fart",
    33: "påbud_gående_syklende",
    34: "påbud_kjør_rett_frem",
    35: "påbud_mot_venstre",
    36: "påbud_rett_frem_eller_høyre",
    37: "påbud_rundkjøring",
    38: "påbud_sykkel",
    39: "påbud_syklende_gående",
    40: "forbud_parkering",
    41: "forbud_stopp",
    42: "forbud_parkering_tid_15",
    43: "forbud_parkering_tid_31",
    44: "info_motgående_viker",
    45: "info_parkering",
    46: "info_parkering_rullestol",
    47: "info_parkering_bil",
    48: "info_parkering_lastebil",
    49: "info_parkering_buss",
    50: "info_parkering_båt",
    51: "info_gatetun",
    52: "info_gatetun_slutt",
    53: "info_kjør_rett_frem",
    54: "info_blindvei",
    55: "info_troll",
    56: "info_fotgjengerfelt",
    57: "info_syklende",
    58: "info_parkering_inn_her",
    59: "info_partshump",
    60: "info_forkjørsvei_slutt",
    61: "info_forkjørsvei"
}

label_to_type_GTSRB = {
    0: "fartsgrense_20",
    1: "fartsgrense_30",
    2: "fartsgrense_50",
    3: "fartsgrense_60",
    4: "fartsgrense_70",
    5: "fartsgrense_80",
    6: "fartsgrense_80_slutt",
    7: "fartsgrense_100",
    8: "fartsgrense_120",
    9: "forbud_forbikjøring",
    10: "fartsgrense_forbikjøring_lastebil",
    11: "fare_forkjørskryss",
    12: "forkjørsveg",
    13: "vikeplikt",
    14: "stopp",
    15: "forbud_blank",
    16: "forbud_lastebil",
    17: "forbud_innkjøring",
    18: "fare_utropstegn",
    19: "fare_venstresving",
    20: "fare_høyresving",
    21: "fare_svingete_venstre",
    22: "fare_dump",
    23: "fare_glatt",
    24: "fare_smal_på_høyre",
    25: "fare_veiarbeid",
    26: "fare_traffiklys",
    27: "fare_fotgjengere",
    28: "fare_forelder_og_barn",
    29: "fare_sykkel",
    30: "fare_snø",
    31: "fare_hjort",
    32: "forbud_blank_slutt",
    33: "påbud_sving_høyre",
    34: "påbud_sving_venstre",
    35: "påbud_kjør_rett_frem",
    36: "påbud_rett_frem_eller_høyre",
    37: "påbud_rett_frem_eller_venstre",
    38: "påbud_hold_høyre",
    39: "påbud_hold_venstre",
    40: "rundkjøring",
    41: "forbud_forbikjøring_slutt",
    42: "forbud_lastebil_slutt"
}

main()