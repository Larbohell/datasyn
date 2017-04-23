import os
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import glob
import time

def classify(test_images, model_dir, input_image_dim):
    # Load training and testing datasets.

    # Restore session and variables/nodes/weights
    session = tf.Session()
    meta_file = os.path.join(model_dir, "save.ckpt.meta")
    saver = tf.train.import_meta_graph(meta_file)

    checkpoint_dir = os.path.join(model_dir)
    saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))

    # Create a graph to hold the model.
    graph = tf.get_default_graph()

    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, input_image_dim[0], input_image_dim[1], 1])

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        weights_0 = tf.global_variables()[0]
        biases_0 = tf.global_variables()[1]
        weights_1 = tf.global_variables()[2]
        biases_1 = tf.global_variables()[3]
        weights_2 = tf.global_variables()[4]
        biases_2 = tf.global_variables()[5]

        hidden1 = tf.nn.relu(tf.matmul(images_flat, weights_0) + biases_0)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_1) + biases_1)
        logits = tf.nn.relu(tf.matmul(hidden2, weights_2) + biases_2)

        predicted_labels = tf.argmax(logits, 1)

    # Run predictions against the full test set.
    start_time = time.time()
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: test_images})[0]
    end_time = time.time()

    with open("timing.txt", "a") as timerfile:
        prediction_time = end_time - start_time
        timerfile.write(prediction_time)
        timerfile.write("\n")

    return predicted
