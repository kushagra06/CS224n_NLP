""" starter code for word2vec skip-gram model with NCE loss
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 04
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

import utils
import word2vec_utils

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016
NUM_VISUALIZE = 2000        # number of tokens to visualize


def word2vec(dataset):
    """ Build the graph for word2vec model and train it """
    # Step 1: create iterator and get input, output from the dataset
    #############################
    ########## TO DO ############
    #############################

    # Step 2: define weights. 
    # In word2vec, it's the weights that we care about
    #############################
    ########## TO DO ############
    #############################

    # Step 3: define the inference (embedding lookup)
    #############################
    ########## TO DO ############
    #############################

    # Step 4: define loss function
    # construct variables for NCE loss
    #############################
    ########## TO DO ############
    #############################

    # define loss function to be NCE loss function
    #############################
    ########## TO DO ############
    #############################

    # Step 5: define optimizer that follows gradient descent update rule
    # to minimize loss
    #############################
    ########## TO DO ############
    #############################
    
    utils.safe_mkdir('checkpoints')
    with tf.Session() as sess:

        # Step 6: initialize iterator and variables
        #############################
        ########## TO DO ############
        #############################

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('graphs/word2vec_simple', sess.graph)

        for index in range(NUM_TRAIN_STEPS):
            try:
                # Step 7: execute optimizer and fetch loss
                #############################
                ########## TO DO ############
                #############################
                loss_batch = None

                total_loss += loss_batch

                if (index + 1) % SKIP_STEP == 0:
                    print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                    total_loss = 0.0
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)
        writer.close()

def gen():
    yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES, VOCAB_SIZE, 
                                        BATCH_SIZE, SKIP_WINDOW, VISUAL_FLD)

def main():
    utils.safe_mkdir('data')
    dataset = tf.data.Dataset.from_generator(gen, 
                                (tf.int32, tf.int32), 
                                (tf.TensorShape([BATCH_SIZE]), tf.TensorShape([BATCH_SIZE, 1])))
    word2vec(dataset)

if __name__ == '__main__':
    main()
