#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import jieba
import nltk
import numpy as np
import collections
from sklearn.model_selection import train_test_split

SEED = 7

def get_data():
    with open('./data.txt', 'r', encoding='utf8') as f:
        data = f.readlines()
        en_sentences = []
        ch_sentences = []
        for line in data:
            [en, ch] = line.strip('\n').split('\t')
            en_sentences.append(en.lower())
            ch_sentences.append(ch)
        del data
        # print(inputs[3465:3475])
        # print(outputs[3465:3475])
        en_sentences = [nltk.word_tokenize(en) for en in en_sentences]
        ch_sentences = [[word for word in jieba.cut(ch, cut_all=False)] for ch in ch_sentences]

        # print(len(max(en_sentences, key=lambda x: len(x))))
        # print(len(max(ch_sentences, key=lambda x: len(x))))

        en_sentences_train, en_sentences_test, ch_sentences_trian, ch_sentences_test = train_test_split(en_sentences, ch_sentences,\
                                                                                                    test_size=0.2, random_state=SEED)
        return en_sentences_train, en_sentences_test, ch_sentences_trian, ch_sentences_test


def build_dict(sentences, init_word=False, max_words=5000):
    word_counts = collections.Counter()
    for sentence in sentences:
        for word in sentence:
            word_counts[word] += 1
    common_words = word_counts.most_common(max_words)
    if init_word:
        total_words = len(common_words) + 4 # ['<UNK>'], ['<PAD>'], ['<GO>'], ['<EOS>']
        word_dict = {word[0]: idx + 4 for idx, word in enumerate(common_words)}
        word_dict['<GO>'] = 2
        word_dict['<EOS>'] = 3
    else:
        total_words = len(common_words) + 2 # ['<UNK>'], ['<PAD>']
        word_dict = {word[0]: idx+2 for idx, word in enumerate(common_words)}
    word_dict['<PAD>'] = 0
    word_dict['<UNK>'] = 1
    # print(word_dict)
    return word_dict, total_words


'''
def add_token(sentences, language='en'):
    for sentence in sentences:
        if language == 'en':
            sentence.insert(0, '<GO>')
        elif language == 'ch':
            sentence.append('<EOS>')
    return sentences
'''


def sentence_to_index(sort_len=True):
    en_sentences_train, en_sentences_test, ch_sentences_trian, ch_sentences_test = get_data()
    en_word_to_index, en_total_words = build_dict(en_sentences_train, init_word=False)
    en_index_to_word = dict([idx, word] for word, idx in en_word_to_index.items())
    ch_word_to_index, ch_total_words = build_dict(ch_sentences_trian, init_word=True)
    ch_index_to_word = {idx: word for word, idx in ch_word_to_index.items()}
    # print(en_word_to_index)
    # print(en_index_to_word)
    # print(ch_word_to_index)
    # print(ch_index_to_word)

    def sorted_by_length(en_sentences, ch_sentences):
        sorted_index = sorted(range(len(en_sentences)), key=lambda x: len(en_sentences[x]), reverse=True)
        en_sentences = [en_sentences[idx] for idx in sorted_index]
        ch_sentences = [ch_sentences[idx] for idx in sorted_index]
        return en_sentences, ch_sentences


    # 将中文和英文按照同样顺序排序
    if sort_len:
        en_sentences_train, ch_sentences_train = sorted_by_length(en_sentences_train, ch_sentences_trian)
        en_sentences_test, ch_sentences_test = sorted_by_length(en_sentences_test, ch_sentences_test)

    encoder_inputs_train = [[en_word_to_index.get(word, en_word_to_index['<UNK>']) for word in sentence] for sentence in en_sentences_train]
    decoder_inputs_train = [[ch_word_to_index['<GO>']] + [ch_word_to_index.get(word, ch_word_to_index['<UNK>']) for word in sentence] for sentence in ch_sentences_train]
    decoder_targets_trian = [[ch_word_to_index.get(word, ch_word_to_index['<UNK>']) for word in sentence] + [ch_word_to_index['<EOS>']] for sentence in ch_sentences_train]

    encoder_inputs_test = [[en_word_to_index.get(word, en_word_to_index['<UNK>']) for word in sentence] for sentence in en_sentences_test]
    decoder_inputs_test = [[ch_word_to_index['<GO>']] + [ch_word_to_index.get(word, ch_word_to_index['<UNK>']) for word in sentence] for sentence in ch_sentences_test]
    decoder_targets_test = [[ch_word_to_index.get(word, ch_word_to_index['<UNK>']) for word in sentence] + [ch_word_to_index['<EOS>']] for sentence in ch_sentences_test]

    return en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words,\
           encoder_inputs_train, decoder_inputs_train, decoder_targets_trian,\
           encoder_inputs_test, decoder_inputs_test, decoder_targets_test


def batch_sentences_to_numpy(sentences):
    sentences_length = np.array([len(sentence) for sentence in sentences])
    max_length = sentences_length.max()
    #<'pad'> == 0
    sentences = [np.array(sentence + [0]*(max_length-len(sentence)) if len(sentence) < max_length else sentence) for sentence in sentences]
    return np.array(sentences), sentences_length


def get_batch(encoder_inputs, decoder_inputs, decoder_targets, batch_size=64):
    batch_num = len(encoder_inputs) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        encoder_inputs_batch, encoder_batch_length = batch_sentences_to_numpy(encoder_inputs[begin:end])
        decoder_inputs_batch, decoder_batch_length = batch_sentences_to_numpy(decoder_inputs[begin:end])
        decoder_targets_batch, _ = batch_sentences_to_numpy(decoder_targets[begin:end])
        # print(type(encoder_inputs_batch))
        # print(encoder_batch_length)
        # print(decoder_batch_length)
        yield encoder_inputs_batch, decoder_inputs_batch, decoder_targets_batch, encoder_batch_length, decoder_batch_length


def generate_predict_sentence(english_sentence, en_word_to_index):
    english_sentence = nltk.word_tokenize(english_sentence)
    encoder_inputs = [en_word_to_index.get(word, en_word_to_index['<UNK>']) for word in english_sentence]
    encoder_inputs, encoder_inputs_length = batch_sentences_to_numpy([encoder_inputs])
    return encoder_inputs, encoder_inputs_length


en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words,\
           encoder_inputs_train, decoder_inputs_train, decoder_targets_trian,\
           encoder_inputs_test, decoder_inputs_test, decoder_targets_test = sentence_to_index(sort_len=True)