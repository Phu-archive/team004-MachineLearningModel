import tensorflow as tf
tf.enable_eager_execution()

from models import positionalEncoding, mask_create, attention, pointwiseFeedForward, encoder, decoder
from models import mask_create
import functools

import re
import numpy as np
import os
import time
import json
from glob import glob
import pickle

import codecs
from io import open
import csv

from utils import extractSentencePairs, loadConversations, loadLines

import pickle

top_k = 7500
num_layers = 4
model_dim = 128
pointWise_dim = 512
num_head = 8
dropout_rate = 0.1
MAX_LENGTH = 20

with open('save_transformer/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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

global_step = tf.train.get_or_create_global_step()
transformer = Transformer(top_k, top_k, model_dim, num_head, dropout_rate, pointWise_dim, num_layers)

def save_tokenizer():
    corpus_name = "dataset/"
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

    all_conversations = []
    all_conversation_sets = []

    with open(datafile, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, lineterminator='\n')
        for i, row in enumerate(reader):
            if i % 20000 == 0:
                print("At ", i)

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

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def getResponse(text, temp = 0.05):
    test_tok = []
    test_tok.append(tokenizer.word_index["<start>"])

    for t in text.split(' '):
        test_tok.append(tokenizer.word_index.get(t, 1))

    test_tok.append(tokenizer.word_index["<end>"])

    while len(test_tok) < 20:
        test_tok.append(0)

    src = tf.convert_to_tensor([test_tok])

    decoder_input = [tokenizer.word_index["<start>"]]
    output = tf.expand_dims(decoder_input, 0)

    def create_mask(inp, target):
        enc_padding_mask = mask_create.create_padding_mask(inp)
        dec_padding_mask = mask_create.create_padding_mask(inp)
        look_ahead_mask = mask_create.create_look_ahead_mask(tf.shape(target)[1])
        dec_target_padding_mask = mask_create.create_padding_mask(target)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(src, output)
    predictions = transformer(src, output, enc_padding_mask, combined_mask, dec_padding_mask, False)

    save_path = "save_transformer/"
    optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.9, beta2=0.98,epsilon=1e-9)

    saver = tf.train.Checkpoint(optimizer=optimizer, model=transformer, optimizer_step=global_step)
    saver.restore(tf.train.latest_checkpoint(save_path))

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_mask(src, output)
        predictions, weights = transformer(src, output, enc_padding_mask, combined_mask, dec_padding_mask, False)
        predictions = predictions[: ,-1:, :]/temp

        predicted_id = tf.multinomial(predictions[0], num_samples=1, output_dtype=tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)
        if predicted_id[0] == tokenizer.word_index["<end>"]:
            break

    reverse = {v:k for k, v in tokenizer.word_index.items()}
    list_text = []
    for a in output.numpy()[0]:
        list_text.append(reverse[a])

        if reverse[a] == '<end>':
            break

    return ' '.join(list_text[1:-1])

def loadTextGetVector(text):
    with open("save_word2vec/types.txt", "r") as f:
        for i, l in enumerate(f.readlines()):
            if text == l:
                break

    with open("save_word2vec/vectors.txt", "r") as f:
        for current_i, l in enumerate(f.readlines()):
            if i == current_i:
                vector = [float(e) for e in l[:-1].split(' ')]
                break

    return vector

# print(loadTextGetVector("fever"))
