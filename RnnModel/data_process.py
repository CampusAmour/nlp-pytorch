#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import random
from torchtext import datasets, data


def get_data(seq_length, seed, split_ratio, vocab_size, batch_size, device):

    tokenize = lambda x: x.split()

    TEXT = data.Field(tokenize=tokenize, lower=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    # print(f'Number of training examples: {len(train_data)}')
    # print(f'Number of testing examples: {len(test_data)}')
    # print(vars(train_data.examples[0]))

    train_data, valid_data = train_data.split(random_state=random.seed(seed), split_ratio=split_ratio)
    # print(len(train_data))
    # print(len(valid_data))

    TEXT.build_vocab(train_data, max_size=vocab_size)
    LABEL.build_vocab(train_data)

    # print(TEXT.vocab.itos[: 10])
    # print(LABEL.vocab.itos)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                               batch_size=batch_size, sort_within_batch=True, device=device)

    # print(next(iter(train_iterator)).label)
    # print(next(iter(train_iterator)).text)
    return TEXT.vocab, train_iterator, valid_iterator, test_iterator