import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from PIL import Image, ImageDraw, ImageFont
import glob

import time

from IPython import get_ipython

TRAINING_NUMBER = 10000
#TRAINING_DATA_SET = "GTSRB"
#TRAINING_DATA_SET = "FromTensorBox/overfeat_rezoom_2017_04_18_23.35"
TRAINING_DATA_SET = "BelgiumTS"
IMAGE_SCALE_SIZE_X = 32
IMAGE_SCALE_SIZE_Y = 32

def main():
    start_time = time.time()
    train()
    end_time = time.time()
    print ("Total time elapsed: ", (end_time-start_time))

# Allow image embeding in notebook
#%matplotlib inline

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

def load_train_data(data_dir):
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

def load_test_data_as_numpy_array(data_dir):
    images = []
    for filename in glob.glob(data_dir+"/*.ppm"):
        images.append(skimage.data.imread(filename)) #Loads the images as a list of numpy arrays

    return images

def save_images_and_labels_to_imagefile(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    #plt.show()
    plt.savefig('labels_and_corresponding_images.png')


def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

def save_model(sess, filename):
    if not os.path.exists("output"): os.makedirs("output")
    dirname = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M')
    os.makedirs(os.path.join("output", filename, dirname))

    saver = tf.train.Saver()
    saver.save(sess, os.path.abspath(os.path.join("output", filename, dirname, "save.ckpt")))
    #saver.save(labels, filename+"_labels")
    # `save` method will call `export_meta_graph` implicitly.
    # you will get saved graph files:my-model.meta


def train():
    # Load training and testing datasets.
    ROOT_PATH = "datasets"
    directory = TRAINING_DATA_SET

    train_data_dir = os.path.join(ROOT_PATH, directory + "/Training")
    #test_data_dir = os.path.join(ROOT_PATH, directory + "/Testing")

    train_images, labels = load_train_data(train_data_dir)
    #test_images = load_test_data_as_numpy_array(test_data_dir)

    #test_images, _ = load_train_data(test_data_dir)

    print("Unique Labels: {0}\nTotal Train Images: {1}".format(len(set(labels)), len(train_images)))
    #print("Total Test Images: ", len(test_images))

    # display_images_and_labels(train_images, labels)

    # display_label_images(train_images, 27)

    # Resize images
    train_images32 = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                      for image in train_images]

    #test_images32 = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
    #                 for image in test_images]

    save_images_and_labels_to_imagefile(train_images32, labels)

    labels_a = np.array(labels)
    train_images_a = np.array(train_images32)
    #test_images_a = np.array(test_images32)
    print("labels: ", labels_a.shape, "\nTrain images: ", train_images_a.shape)# "\nTest images: ", test_images_a.shape)

    # Create a graph to hold the model.
    graph = tf.Graph()

    # Create model in the graph.
    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y, 3], name="images_ph")
        labels_ph = tf.placeholder(tf.int32, [None])

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
        #tf.add_to_collection("logits", logits)
        #print("images_ph: ", images_ph)
        #tf.add_to_collection("images_ph", images_ph)

        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)

        # Define the loss function.
        # Cross-entropy is a good choice for classification.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

        # Create training op.
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # And, finally, an initialization op to execute before training.
        init = tf.global_variables_initializer()

        print("images_flat: ", images_flat)
        print("logits: ", logits)
        print("loss: ", loss)
        print("predicted_labels: ", predicted_labels)
        print("images_ph: ", images_ph)

        # Create a session to run the graph we created.
        session = tf.Session(graph=graph)

        # First step is always to initialize all variables.
        # We don't care about the return value, though. It's None.
        _ = session.run([init])

        #The actual training
        start = time.time()
        for i in range(TRAINING_NUMBER):
            _, loss_value = session.run([train, loss],
                                        feed_dict={images_ph: train_images_a, labels_ph: labels_a})
            if i % 50 == 0:
                print("Iter: " + str(i) +", Loss: ", loss_value, ", Time elapsed: ", time.time()-start)

        #TESTING
        #predicted_labels = session.run([predicted_labels],
        #                               feed_dict={images_ph: test_images32})[0]

        #EVALUATING THE TEST
        #save_dir = 'output/' + directory + '/predictions'
        #if not os.path.exists(save_dir):
        #    os.makedirs(save_dir)

        #i = 0

        #for pl in predicted_labels:
        #    predicted_image = test_images32[i]
        #    save_numpy_array_as_image(predicted_image,save_dir,'/label_'+str(pl)+'_'+str(i)+'.png')
            #p_i.save(save_dir+'/label_'+str(pl)+'_'+str(i)+'.png')
        #    i+=1

        # # Pick 10 random train images
        # sample_indexes = random.sample(range(len(train_images32)), 10)
        # sample_images = [train_images32[i] for i in sample_indexes]
        # sample_labels = [labels[i] for i in sample_indexes]
        #
        # # Run the "predicted_labels" op.
        # predicted = session.run([predicted_labels],
        #                         feed_dict={images_ph: sample_images})[0]
        #
        # print(sample_labels)
        # p = "["
        # for i in range(len(predicted)):
        #     p += str(predicted[i]) + ", "
        # print(p)
        #
        # # Display the predictions and the ground truth visually.
        # fig = plt.figure(figsize=(10, 10))
        # for i in range(len(sample_images)):
        #     truth = sample_labels[i]
        #     prediction = predicted[i]
        #     plt.subplot(5, 2, 1 + i)
        #     plt.axis('off')
        #     color = 'green' if truth == prediction else 'red'
        #     plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
        #              fontsize=12, color=color)
        #     plt.imshow(sample_images[i])
        # plt.show()

        print("Saving now...")
        print("logits: ", logits)
        print("predicted_labels: ", predicted_labels)
        print("images_ph: ", images_ph)

        # Save session
        save_model(session, directory)
        # Close the session. This will destroy the trained model.
        session.close()

main()