import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

TRAINING_TEST_DATA = 2
MODEL_FOLDER_PATH = "BelgiumTS/2017_04_20_13.47" #"BelgiumTS/2017_04_18_17.57"

IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    # Load training and testing datasets.
    ROOT_PATH = "datasets"
    directory = "GTSRB"
    if (TRAINING_TEST_DATA == 2):
        directory = "BelgiumTS"
    if (TRAINING_TEST_DATA == 1):
        directory = "Training_test_small_set"
    #train_data_dir = os.path.join(ROOT_PATH, directory, "Training")
    test_data_dir = os.path.join(ROOT_PATH, directory, "Testing")

    # Load session
    #session = restore_model(MODEL_FOLDER_PATH)

    session = tf.Session()
    # labels = tf.Tensor()
    meta_file = os.path.join("output", MODEL_FOLDER_PATH, "save.ckpt.meta")
    new_saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join("output", MODEL_FOLDER_PATH)

    new_saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

    for v in tf.global_variables():
        print(v.name)
        session.run(v)
    #tf.variables_initializer(tf.global_variables())

    #session.run(tf.global_variables_initializer())

    # Load the test dataset.
    test_images, test_labels = load_data(test_data_dir)

    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                     for image in test_images]
    # display_images_and_labels(test_images32, test_labels)

    #tf.reset_default_graph()
    images_ph = tf.placeholder(tf.float32, [None, IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y, 3])

    # Create a graph to hold the model.
    graph = session.graph
    #graph = tf.Graph()

    # Create model in the graph.

    with graph.as_default():
        # # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y, 3])
        #labels_ph = tf.placeholder(tf.int32, [None])
        #
        # # Flatten input from: [None, height, width, channels]
        # # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)
        #
        # # Fully connected layer.
        # # Generates logits of size [None, 62]
        logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        #logits = tf.get_collection("logits")[0]
        #logitss = tf.get_variable("logitssss", shape=62)
        #print(logits)
        #images_ph = tf.get_collection("images_ph")
        #print(images_ph)

        predicted_labels = tf.argmax(logits, 1)
        #print(predicted_labels)

    session.run(tf.global_variables_initializer())

    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]


    # Calculate how many matches we got.
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))

    # EVALUATING THE TEST
    save_dir = 'output/' + directory + '/predictions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    i = 0

    for pl in predicted:
        predicted_image = test_images32[i]
        save_numpy_array_as_image(predicted_image, save_dir, '/label_' + str(pl) + '_' + str(i) + '.png')
        # p_i.save(save_dir+'/label_'+str(pl)+'_'+str(i)+'.png')
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

def restore_model(sess, model_folder_name):
    sess = tf.Session()
    #labels = tf.Tensor()
    meta_file = os.path.join("output", model_folder_name, "save.meta")
    new_saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join("output", model_folder_name)
    sess.run(tf.global_variables_initializer())
    new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)

    #new_saver = tf.train.import_meta_graph(filename + "_labels" + '.meta')
    #new_saver.restore(labels, tf.train.latest_checkpoint('./'))
    return sess#, labels

main()