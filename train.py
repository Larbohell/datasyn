import os
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import glob
import time

TRAINING_NUMBER = 100
DISPLAY_FREQUENCY = 10
MODEL_DIR = "BelgiumTS/2017_04_21_22.05_1"
CONTINUE_TRAINING_ON_MODEL = True

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
    dirname = datetime.datetime.now().strftime('%Y_%m_%d_%H.%M') + "_" + str(TRAINING_NUMBER)
    os.makedirs(os.path.join("output", filename, dirname))

    saver = tf.train.Saver()
    saver.save(sess, os.path.abspath(os.path.join("output", filename, dirname, "save.ckpt")))
    # `save` method will call `export_meta_graph` implicitly.
    # you will get saved graph files:my-model.meta


def train():
    # Load training and testing datasets.
    ROOT_PATH = "datasets"
    directory = TRAINING_DATA_SET

    train_data_dir = os.path.join(ROOT_PATH, directory + "/Training")
    train_images, labels = load_train_data(train_data_dir)

    print("Unique Labels: {0}\nTotal Train Images: {1}".format(len(set(labels)), len(train_images)))

    # display_images_and_labels(train_images, labels)

    # display_label_images(train_images, 27)

    # Resize images
    train_images32 = [skimage.transform.resize(image, (IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y))
                      for image in train_images]

    save_images_and_labels_to_imagefile(train_images32, labels)

    labels_a = np.array(labels)
    train_images_a = np.array(train_images32)
    print("labels: ", labels_a.shape, "\nTrain images: ", train_images_a.shape)

    if CONTINUE_TRAINING_ON_MODEL:
        # Restore session and variables/nodes/weights
        session = tf.Session()
        meta_file = os.path.join("output", MODEL_DIR, "save.ckpt.meta")
        saver = tf.train.import_meta_graph(meta_file)

        checkpoint_dir = os.path.join("output", MODEL_DIR)
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))
        graph = tf.get_default_graph()
    else:
        # Create a graph to hold the model.
        graph = tf.Graph()

    # Create model in the graph.
    with graph.as_default():
        # Placeholders for inputs and labels.
        images_ph = tf.placeholder(tf.float32, [None, IMAGE_SCALE_SIZE_X, IMAGE_SCALE_SIZE_Y, 3],
                                   name="images_ph")
        labels_ph = tf.placeholder(tf.int32, [None])

        # Flatten input from: [None, height, width, channels]
        # To: [None, height * width * channels] == [None, 3072]
        images_flat = tf.contrib.layers.flatten(images_ph)

        # Fully connected layer.
        # Generates logits of size [None, 62]
        if CONTINUE_TRAINING_ON_MODEL:
            weights_0 = tf.global_variables()[0]
            biases_0 = tf.global_variables()[1]
            weights_1 = tf.global_variables()[2]
            biases_1 = tf.global_variables()[3]
            weights_2 = tf.global_variables()[4]
            biases_2 = tf.global_variables()[5]

            hidden1 = tf.nn.relu(tf.matmul(images_flat, weights_0) + biases_0)
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_1) + biases_1)
            logits = tf.nn.relu(tf.matmul(hidden2, weights_2) + biases_2)

            #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))
            loss = graph.get_tensor_by_name("Loss")
            train = graph.get_operation_by_name("Adam")
            print(train)
            #train = adam.minimize(loss)
        else:
            hidden1 = tf.contrib.layers.fully_connected(images_flat, 100, tf.nn.relu)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, 100, tf.nn.relu)
            logits = tf.contrib.layers.fully_connected(hidden2, 62, tf.nn.relu)

            # Define the loss function.
            # Cross-entropy is a good choice for classification.
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph), name = "Loss")
            # Create training op.
            adam = tf.train.AdamOptimizer(learning_rate=0.001, name = "Adam")
            train = adam.minimize(loss)
            print(train)

        # Convert logits to label indexes (int).
        # Shape [None], which is a 1D vector of length == batch_size.
        predicted_labels = tf.argmax(logits, 1)

        print("images_flat: ", images_flat)
        print("logits: ", logits)
        print("loss: ", loss)
        print("predicted_labels: ", predicted_labels)
        print("images_ph: ", images_ph)

        if not CONTINUE_TRAINING_ON_MODEL:
            # And, finally, an initialization op to execute before training.
            init = tf.global_variables_initializer()

            # Create a session to run the graph we created.
            session = tf.Session(graph=graph)

            # First step is always to initialize all variables.
            # We don't care about the return value, though. It's None.
            _ = session.run([init])

        #The actual training
        start = time.time()
        loss_value = None

        for i in range(TRAINING_NUMBER):
            _, loss_value = session.run([train, loss],
                                        feed_dict={images_ph: train_images_a, labels_ph: labels_a})
            if i % DISPLAY_FREQUENCY == 0:
                print("Iter: " + str(i) +", Loss: ", loss_value, ", Time elapsed: ", time.time()-start)

        print("Iter: " + str(TRAINING_NUMBER-1) + ", Loss: ", loss_value, ", Time elapsed: ", time.time() - start)

       # Save session
        save_model(session, directory)
        # Close the session. This will destroy the trained model.
        session.close()
        print("Model saved.")

main()