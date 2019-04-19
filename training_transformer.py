import tensorflow as tf
tf.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.enable_eager_execution(config=config)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import re
import numpy as np
import os
import time
import json
from glob import glob
import pickle
import argparse

import math
import codecs
from io import open
import csv

import functools
import sys

from models import positionalEncoding, mask_create, attention, pointwiseFeedForward, encoder, decoder
from models import mask_create

from utils import extractSentencePairs, loadConversations, loadLines

def get_learning_rate(model_size, warmup):

    def decayed_lr(model_size, warmup):
        current_learning_rate = (model_size ** (-0.5) * \
                                    min(tf.train.get_global_step().numpy() ** (-0.5),
                                    tf.train.get_global_step().numpy() * warmup ** (-1.5)) )
        return current_learning_rate

    return functools.partial(decayed_lr, model_size, warmup)

def calculate_loss(target, pred):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=pred)
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)

def create_mask(inp, target):
    enc_padding_mask = mask_create.create_padding_mask(inp)
    dec_padding_mask = mask_create.create_padding_mask(inp)
    look_ahead_mask = mask_create.create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = mask_create.create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

class Transformer(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer):
        super().__init__()

        self.encoder = encoder.Encoder(input_vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer)
        self.decoder = decoder.Decoder(target_vocab_size, model_dim, num_head, drop_prob, pointWise_dim, num_sublayer)

    def call(self, x, target, mask, look_ahead_mask, padding_mask, is_training):
        encoded, weight_encoder = self.encoder(
            x,
            mask=mask,
            is_training=is_training
        )

        out, weight_decoder = self.decoder(
            encoded,
            target,
            look_ahead_mask,
            padding_mask,
            is_training
        )
        return out, (weight_encoder, weight_decoder)

def train_step(transformer, optimizer, inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, enc_padding_mask, combined_mask, dec_padding_mask, True)
        loss = calculate_loss(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss


def main(argv):
    argparser = argparse.ArgumentParser('Transformer-Chatbot', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--vocabulary-size', action='store', type=int, default=7500)
    argparser.add_argument('--max-sen-length', action='store', type=int, default=20)
    argparser.add_argument('--num-layer', action='store', type=int, default=4)
    argparser.add_argument('--model-dim', action='store', type=int, default=128)
    argparser.add_argument('--pointwise-dim', action='store', type=int, default=512)
    argparser.add_argument('--num-head', action='store', type=int, default=8)
    argparser.add_argument('--dropout-rate', action='store', type=float, default=0.1)
    argparser.add_argument('--batch-size', action='store', type=int, default=64)
    argparser.add_argument('--warmup-length', action='store', type=int, default=4000)

    args = argparser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    top_k = args.vocabulary_size
    MAX_LENGTH = args.max_sen_length
    num_layers = args.num_layer
    model_dim = args.model_dim
    pointWise_dim = args.pointwise_dim
    num_head = args.num_head
    dropout_rate = args.dropout_rate
    BATCH_SIZE = args.batch_size
    warmup = args.warmup_length
    BUFFER_SIZE = 1000

    corpus_name = "./dataset/"
    corpus = os.path.join(corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)

    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)

    all_conversations = []
    all_conversation_sets = []

    with open(datafile, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, lineterminator='\n')
        for i, row in enumerate(reader):
            if i % 20000 == 0:
                print(f"At {i}")

            start = "<start> " + row[0] + " <end>"
            end = "<start> " + row[1] + " <end>"

            all_conversation_sets.append((
                start,
                end
            ))
            all_conversations.append(start)
            all_conversations.append(end)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(all_conversations)
    tokenizer.word_index['<pad>'] = 0
    train_seqs = tokenizer.texts_to_sequences(all_conversations)

    train_seqs_inp, train_seqs_out  = [], []
    for i in range(len(train_seqs)//2):
        first = i*2
        second = first + 1
        if len(train_seqs[first]) > MAX_LENGTH or len(train_seqs[second]) > MAX_LENGTH:
            continue

        train_seqs_inp.append(train_seqs[first])
        train_seqs_out.append(train_seqs[second])

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_seqs_inp, padding='post')
    out_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_seqs_out, padding='post')
    vocab_size = len(tokenizer.word_index)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, out_tensor))
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    global_step = tf.train.get_or_create_global_step()
    transformer = Transformer(top_k, top_k, model_dim, num_head, dropout_rate, pointWise_dim, num_layers)

    optimizer = tf.train.AdamOptimizer(get_learning_rate(model_dim, warmup), beta1=0.9, beta2=0.98,
                                         epsilon=1e-9)

    saver = tf.train.Checkpoint(optimizer=optimizer, model=transformer, optimizer_step=global_step)
    summary_writer = tf.contrib.summary.create_file_writer(f"summary/transformer")


    epoch = 0
    for epoch in range(20):
        start = time.time()
        for (batch, (inp, tar)) in enumerate(dataset):
            loss = train_step(transformer, optimizer, inp, tar)
            if batch%100 == 0:
                print(f"At {batch}/11724 the loss is {loss}")
                with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar("loss", loss)
            global_step.assign_add(1)

        end = time.time()
        print(f"Take around for one epoch {end - start}")
        sys.stdout.flush()

        if not os.path.exists('save'):
            os.mkdir('save')

        saver.save(f"save/transformer_{epoch}.ckpt")

if __name__ == '__main__':
    main(sys.argv[1:])
