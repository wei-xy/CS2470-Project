from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 20
        self.num_classes = 3
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main

        # two hidden (dense) layer sizes
        self.h1 = 200
        self.h2 = 100

        # Initializes all trainable parameters
        # will use 3 convolution layers and 3 dense layers
        # will initialize W1 assuming 'SAME' padding is used
        self.filter1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 16], stddev=0.1))
        self.bias1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))

        self.filter2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=0.1))
        self.bias2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

        self.filter3 = tf.Variable(tf.random.truncated_normal([3, 3, 20, 20], stddev=0.1))
        self.bias3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

        self.dense1 = tf.keras.layers.Dense(units=self.h1, use_bias=True, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=self.h2, use_bias=True, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=self.num_classes, use_bias=True, activation='relu')

    def call(self, inputs, is_testing=True):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: adjacency matrices, shape of (num_inputs, 100, 100, 1)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes)
        """

        # need to expand dims to add a "channel" of size 1
        inputs = tf.expand_dims(inputs, axis=-1)  # to be consistent with a channel size of 1 (for images)

        # layer 1
        layer1Output = tf.nn.conv2d(inputs, self.filter1, strides=[1, 1, 1, 1], padding='SAME')
        layer1Output = tf.nn.bias_add(layer1Output, self.bias1)
        mean, var = tf.nn.moments(layer1Output, axes=[0, 2, 1])
        layer1Output = tf.nn.batch_normalization(layer1Output, mean, var, scale=1, offset=0, variance_epsilon=1e-5)
        layer1Output = tf.nn.leaky_relu(layer1Output)
        layer1Output = tf.nn.max_pool(layer1Output, 3, strides=2, padding='SAME')

        # layer 2
        layer2Output = tf.nn.conv2d(layer1Output, self.filter2, strides=[1, 1, 1, 1], padding='SAME')
        layer2Output = tf.nn.bias_add(layer2Output, self.bias2)
        mean, var = tf.nn.moments(layer2Output, axes=[0, 2, 1])
        layer2Output = tf.nn.batch_normalization(layer2Output, mean, var, scale=1, offset=0, variance_epsilon=1e-5)
        layer2Output = tf.nn.leaky_relu(layer2Output)
        layer2Output = tf.nn.max_pool(layer2Output, 3, strides=2, padding='SAME')

        # layer 3
        layer3Output = tf.nn.conv2d(layer2Output, self.filter3, strides=[1, 1, 1, 1], padding='SAME')
        layer3Output = tf.nn.bias_add(layer3Output, self.bias3)
        mean, var = tf.nn.moments(layer3Output, axes=[0, 2, 1])
        layer3Output = tf.nn.batch_normalization(layer3Output, mean, var, scale=1, offset=0, variance_epsilon=1e-5)
        layer3Output = tf.nn.leaky_relu(layer3Output)
        layer3Output = tf.nn.max_pool(layer3Output, 3, strides=2, padding='SAME')

        # flattening into shape = (num_examples, image_sz x image_sz x last_filter_sz)
        layer3Output = tf.reshape(layer3Output, [layer3Output.shape[0], -1])

        # dense layer 1
        denseLayer1 = self.dense1(layer3Output)
        denseLayer1 = tf.nn.dropout(denseLayer1, rate=0.01)
        denseLayer1 = tf.nn.leaky_relu(denseLayer1, alpha=0.2)

        # dense layer 2
        denseLayer2 = self.dense2(denseLayer1)
        denseLayer2 = tf.nn.dropout(denseLayer2, rate=0.01)
        denseLayer2 = tf.nn.leaky_relu(denseLayer2, alpha=0.2)

        # dense layer 3
        denseLayer3 = self.dense3(denseLayer2)
        denseLayer3 = tf.nn.dropout(denseLayer3, rate=0.01)
        denseLayer3 = tf.nn.leaky_relu(denseLayer3, alpha=0.2)

        return denseLayer3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # defines a global instance of the optimizer, to be used in train
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    input_sz = train_inputs.shape[0]
    for b in range(0, input_sz, model.batch_size):
        images_in_batch = train_inputs[b: b + model.batch_size]
        labels_in_batch = train_labels[b: b + model.batch_size]

        # randomly flip images in batch
        images_in_batch = tf.image.random_flip_left_right(images_in_batch)

        with tf.GradientTape() as tape:
            logits = model.call(images_in_batch)  # calls the model on a batch
            loss = model.loss(logits, labels_in_batch)
            # keeping track of training accuracy:
            if b // model.batch_size % 5 == 0:
                train_acc = model.accuracy(model(images_in_batch), labels_in_batch)
                print("Accuracy on training set after {} training steps: {}".format(b, train_acc))

        # this section taken from the lab
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def get_data(file_location):
    """
    geta_data will unpickle the data which will have been pickled as a tuple (graphs, labels)
    """
    with open(file_location, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    logits = model.call(test_inputs)
    accuracy = model.accuracy(logits, test_labels)

    print('model accuracy is', accuracy)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def main():
    '''
     - reads in graph data (1000 graphs of each type)
         - divides data into 80% train and 20% test
     - trains the network model
     - tests the model
     - returns the accuracy
    '''

    train_graphs, train_labels = get_data('geom_graphs')

    model = Model()

    # trains the model for 10 epochs
    for epoch in range(10):
        print('Epoch number', epoch)

        shuffled_indices = tf.random.shuffle(np.arange(train_inputs.shape[0]))
        shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
        shuffled_labels = tf.gather(train_labels, shuffled_indices)

        train(model, train_inputs=shuffled_inputs, train_labels=shuffled_labels)

    test(model, test_inputs, test_labels)


if __name__ == '__main__':
    main()
