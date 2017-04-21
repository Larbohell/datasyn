import os
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime

MODEL_DIR = "BelgiumTS/2017_04_21_18.51" #"BelgiumTS/2017_04_18_17.57"
TEST_DATA_SET = "BelgiumTS"
#TEST_DATA_SET = "GTSRB"
#TEST_DATA_SET = "Training_test_small_set"

IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    # Load training and testing datasets.
    ROOT_PATH = "datasets"
    directory = TEST_DATA_SET
    test_data_dir = os.path.join(ROOT_PATH, directory, "Testing")

    # Restore session and variables/nodes/weights
    session = tf.Session()
    meta_file = os.path.join("output", MODEL_DIR, "save.ckpt.meta")
    saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join("output", MODEL_DIR)
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

    # Load the test dataset.
    test_images, test_labels = load_data(test_data_dir)

    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                     for image in test_images]

    # Create a graph to hold the model.
    graph = tf.get_default_graph()

    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y, 3])

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        weights = tf.global_variables()[0]
        biases = tf.global_variables()[1]
        logits = tf.nn.relu(tf.matmul(images_flat, weights) + biases)

        predicted_labels = tf.argmax(logits, 1)

    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]

    # Calculate how many matches we got.
    if (len(test_labels) != len(predicted)):
        print("Length of test labels != length og predicted and accuracy is not correctly calculated")
        print("Length of test labels: ", len(test_labels))
        print("Lenght of predicted: ", len(predicted))
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))

    # EVALUATING THE TEST
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')
    save_dir = 'output/' + directory + '/predictions_' + timestamp

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0

    for pl in predicted:
        predicted_image = test_images32[i]
        save_numpy_array_as_image(predicted_image, save_dir, '/label_' + str(pl) + '_' + str(i) + '.png')
        i += 1

def save_numpy_array_as_image(array, save_dir, filename):
    #Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / array.max() * (array - array.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(save_dir + filename)

def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

main()