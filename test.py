import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TRAINING_TEST_DATA = 2

def main():
    # Load training and testing datasets.
    ROOT_PATH = "datasets"
    directory = "GTSRB"
    if (TRAINING_TEST_DATA == 2):
        directory = "BelgiumTS"
    train_data_dir = os.path.join(ROOT_PATH, directory + "/Training")
    test_data_dir = os.path.join(ROOT_PATH, directory + "/Testing")

    # Load session
    session = restore_model(directory+"_model")

    # Load the test dataset.
    test_images, test_labels = load_data(test_data_dir)

    # Transform the images, just like we did with the training set.
    test_images32 = [skimage.transform.resize(image, (32, 32))
                     for image in test_images]
    # display_images_and_labels(test_images32, test_labels)

    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images32})[0]
    # Calculate how many matches we got.
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))

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

def restore_model(filename):
    sess = tf.Session()
    labels = tf.Tensor()
    new_saver = tf.train.import_meta_graph(filename+"_model"+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    all_vars = tf.get_collection('vars')
    for v in all_vars:
        v_ = sess.run(v)
        print(v_)

    new_saver = tf.train.import_meta_graph(filename + "_labels" + '.meta')
    new_saver.restore(labels, tf.train.latest_checkpoint('./'))
    return sess, labels

main()