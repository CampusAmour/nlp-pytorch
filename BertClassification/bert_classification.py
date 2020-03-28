#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import torch.nn as nn
import numpy as np
import time
from transformers import BertTokenizer, BertModel, BertConfig

MODEL_PATH = './bert-base-chinese'
VOCAB_PATH = MODEL_PATH + '/bert-base-chinese-vocab.txt'
BATCH_SIZE = 64
EPOCHS = 100
FILE_PATH = './THUCNews/data/'
LEARNING_RATE = 1e-4
SAVE_MODEL_PATH = './bert_model/bert_model.pth'
MAX_SEQ_LENGTH = 36
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)

def generate_text_classes(file_path):
    with open(file_path, 'r') as f:
        return f.read().split('\n')


def generate_data(file_path, max_seq_length):

    data_set, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            data, label = line.rstrip('\n').split('\t')
            tokenized_data = tokenizer.tokenize(data)
            tokenized_data.insert(0, "[CLS]")
            tokenized_data.append("[SEP]")
            # print(tokenized_data)

            # 将 token 转为 vocabulary 索引
            data_indexed = tokenizer.convert_tokens_to_ids(tokenized_data)
            # print(data_indexed)

            data_pad = [tokenizer.pad_token_id] * max_seq_length

            seq_length = min(len(data_indexed), max_seq_length)
            data_pad[:seq_length] = data_indexed[:seq_length]

            data_set.append(np.array(data_pad))
            labels.append(int(label))
        data_set = torch.from_numpy(np.array(data_set))

    # print('data_set:', len(data_set))
    # print('labels:', len(labels))
    return data_set.to(device), torch.from_numpy(np.array(labels)).to(device)


def get_batch(data, labels, batch_size=64):
    batch_num = len(data) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        data_batch = data[begin: end]
        labels_batch = labels[begin: end]
        yield data_batch, labels_batch


class BertClassifyModel(nn.Module):
    def __init__(self, config, num_class, fc_dropout=0.2):
        super(BertClassifyModel, self).__init__()
        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained(MODEL_PATH)

        # 锁梯度,做迁移学习
        for param in self.bert.parameters():
            param.requires_grad = False
            # print(param.requires_grad)

        self.fc = nn.Linear(config.hidden_size, num_class)
        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x):
        # bert return: (last_hidden_​​state, pooler_output)
        _, outputs = self.bert(x)
        return self.dropout(self.fc(outputs))


classes = generate_text_classes(FILE_PATH + 'class.txt')


def evaluate(model, eval_data, eval_labels, criterion, batch_size):
    total_loss = 0.
    batch_num = len(eval_data) // BATCH_SIZE
    batch = get_batch(eval_data, eval_labels, batch_size)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            eval_batch, eval_labels = next(batch)

            predict = model(eval_batch)

            predict = predict.view(-1, predict.shape[-1])  # [batch_size, classes]
            loss = criterion(predict, eval_labels)
            total_loss += (loss.cpu().item() * BATCH_SIZE)
    return total_loss / (batch_num * BATCH_SIZE)


def test(model, test_data, test_labels, batch_size):
    acc_count = 0
    batch_num = len(test_data) // BATCH_SIZE
    batch = get_batch(test_data, test_labels, batch_size)
    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i in range(batch_num):
            test_batch, test_labels = next(batch)

            predict = model(test_batch)
            predict = torch.argmax(softmax(predict), dim=1)

            acc_count += torch.sum(torch.Tensor([x == y for x, y in zip(predict, test_labels)])).long().cpu().item()
    return acc_count / (batch_num * BATCH_SIZE)


def train():
    train_data, train_labels = generate_data(FILE_PATH + 'train.txt', MAX_SEQ_LENGTH)
    test_data, test_labels = generate_data(FILE_PATH + 'test.txt', MAX_SEQ_LENGTH)
    eval_data, eval_labels = generate_data(FILE_PATH + 'dev.txt', MAX_SEQ_LENGTH)

    config = BertConfig()
    model = BertClassifyModel(config, len(classes), DROPOUT)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.time()
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.
            batch_num = len(train_data) // BATCH_SIZE
            batch = get_batch(train_data, train_labels, BATCH_SIZE)
            for i in range(batch_num):
                data_batch, labels_batch = next(batch) # labels_batch: [batch_size]

                outputs = model(data_batch)

                outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size, class]

                optimizer.zero_grad()
                loss = criterion(outputs, labels_batch)
                total_loss += (loss.cpu().item() * BATCH_SIZE)

                loss.backward()
                optimizer.step()

            eval = evaluate(model, eval_data, eval_labels, criterion, BATCH_SIZE)

            # accuracy_train = test(model, train_data, train_labels, BATCH_SIZE)
            accuracy_eval = test(model, eval_data, eval_labels, BATCH_SIZE)
            accuracy_test = test(model, test_data, test_labels, BATCH_SIZE)


            print('epoch %d, loss_train %.4f, loss_eval %.4f, accuracy_eval %.4f, accuracy_test %.4f, time %.2fmin' %\
                  (epoch + 1, total_loss / (batch_num * BATCH_SIZE), eval, accuracy_eval, accuracy_test, (time.time() - start) / 60))

        torch.save(model.state_dict(), SAVE_MODEL_PATH)
    except KeyboardInterrupt:
        # ctrl + c
        print('检测到外部中断,训练结束,模型已自动保存~')
        path = './bert_model/epoch_' + str(epoch) + '_epochbert_model.pth'
        torch.save(model.state_dict(), path)


def predict():
    # data = input('请输入测试数据:')
    data = '四川成都发布中小学开学时间表'
    print(data)
    tokenized_data = tokenizer.tokenize(data)
    tokenized_data.insert(0, "[CLS]")
    tokenized_data.append("[SEP]")
    data_indexed = tokenizer.convert_tokens_to_ids(tokenized_data)
    data = torch.from_numpy(np.array(data_indexed)).to(device)
    data = data.unsqueeze(0) # [1, seq_length]

    config = BertConfig()
    model = BertClassifyModel(config, len(classes), DROPOUT)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)

    print(classes)
    with torch.no_grad():

        predict = model(data)
        predict_softmax = softmax(predict)
        print(predict_softmax)
        predict = torch.argmax(predict_softmax, dim=1)
        print(classes[predict.cpu().item()])


if __name__ == '__main__':
    train()
    # predict()