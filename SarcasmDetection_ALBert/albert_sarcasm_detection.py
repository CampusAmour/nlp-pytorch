#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import torch.nn as nn
import numpy as np
import time
import random
from transformers import AlbertModel, AlbertTokenizer, AlbertConfig
from sklearn.metrics import f1_score, accuracy_score

random.seed(7)
MODEL_PATH = './albert-base-v2'
BATCH_SIZE = 32
EPOCHS = 100
FILE_PATH = './data/'
LEARNING_RATE = 1e-5
SAVE_MODEL_PATH = './albert_model/albert_model.pth'
MAX_SEQ_LENGTH = 36
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
MAX_SEQ_LENGTH = 75


def generate_data(file_path, max_seq_length):
    with open(file_path, 'r') as f:
        total_data = f.readlines()
    random.shuffle(total_data)

    sentences, labels = [], []

    for data in total_data:
        _, label, sentence= data.rstrip('\n').split('\t')
        labels.append(int(label))
        tokenized_sentence = tokenizer.tokenize(sentence)
        tokenized_sentence.insert(0, "[CLS]")
        tokenized_sentence.append("[SEP]")
        if len(tokenized_sentence) < max_seq_length:
            tokenized_sentence += ["PAD"] * (max_seq_length - len(tokenized_sentence))
        sentence_indexed = tokenizer.convert_tokens_to_ids(tokenized_sentence)
        sentences.append(np.array(sentence_indexed))
    return torch.from_numpy(np.array(sentences)), torch.from_numpy(np.array(labels))


def generate_batch(sentences, labels, batch_size=64):
    batch_num = len(sentences) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        sentences_batch = sentences[begin: end]
        labels_batch = labels[begin: end]
        yield sentences_batch.to(device), labels_batch.to(device)


class ALBertClassifyModel(nn.Module):
    def __init__(self, config, num_class, fc_dropout=0.1):
        super(ALBertClassifyModel, self).__init__()
        # self.bert = BertModel(config)
        self.albert = AlbertModel.from_pretrained(MODEL_PATH)

        # 锁梯度,做迁移学习
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        #     # print(param.requires_grad)
        self.fc = nn.Linear(config.hidden_size, num_class)
        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x):
        # albert return: (last_hidden_​​state, pooler_output)
        _, outputs = self.albert(x)
        return self.dropout(self.fc(outputs))


def test(model, test_data, test_labels, batch_size):
    batch_num = len(test_data) // BATCH_SIZE
    batch = generate_batch(test_data, test_labels, batch_size)
    model.eval()
    softmax = nn.Softmax(dim=1)
    predicts = []
    with torch.no_grad():
        for i in range(batch_num):
            test_batch, _ = next(batch)

            predict = model(test_batch)
            predict = torch.argmax(softmax(predict), dim=1)
            predicts.extend(list(predict.cpu().numpy()))

    return f1_score(test_labels.cpu().numpy()[:batch_num*BATCH_SIZE], np.array(predicts), average='micro'), \
           accuracy_score(test_labels.numpy()[:batch_num * BATCH_SIZE], np.array(predicts))


def evaluate(model, eval_data, eval_labels, criterion, batch_size):
    total_loss = 0.
    batch_num = len(eval_data) // BATCH_SIZE
    batch = generate_batch(eval_data, eval_labels, batch_size)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            eval_batch, eval_labels = next(batch)

            predict = model(eval_batch)

            predict = predict.view(-1, predict.shape[-1])  # [batch_size, classes]
            loss = criterion(predict, eval_labels)
            total_loss += (loss.cpu().item() * BATCH_SIZE)
    return total_loss / (batch_num * BATCH_SIZE)


def train():
    train_sentences, train_labels = generate_data(FILE_PATH+'Train_v1.txt', MAX_SEQ_LENGTH)
    test_sentences, test_labels = generate_data(FILE_PATH + 'Test_v1.txt', MAX_SEQ_LENGTH)

    config = AlbertConfig(hidden_size=768)
    model = ALBertClassifyModel(config, num_class=2, fc_dropout=DROPOUT)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.time()
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.
            batch_num = len(train_sentences) // BATCH_SIZE
            batch = generate_batch(train_sentences, train_labels, BATCH_SIZE)
            for i in range(batch_num):
                data_batch, labels_batch = next(batch) # labels_batch: [batch_size]

                outputs = model(data_batch)

                outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size, class]

                optimizer.zero_grad()
                loss = criterion(outputs, labels_batch)
                total_loss += (loss.cpu().item() * BATCH_SIZE)

                loss.backward()
                optimizer.step()

            f1_score_test, accuracy_test = test(model, test_sentences, test_labels, BATCH_SIZE)

            print('epoch %d, loss_train %.4f, accuracy_test %.4f, f1_score_test % .4f, time %.2fmin' % \
                  (epoch+1, total_loss/(batch_num*BATCH_SIZE), accuracy_test, f1_score_test, (time.time()-start)/60))
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
    except KeyboardInterrupt:
        # ctrl + c
        print('检测到外部中断,训练结束,模型已自动保存~')
        path = './albert_model/epoch_' + str(epoch) + '_epochbert_model.pth'
        torch.save(model.state_dict(), path)


def prediction():
    # data = input('请输入测试数据:')
    data = "Don't give me your attitude!"
    print(data)
    tokenized_data = tokenizer.tokenize(data)
    tokenized_data.insert(0, "[CLS]")
    tokenized_data.append("[SEP]")
    data_indexed = tokenizer.convert_tokens_to_ids(tokenized_data)
    data = torch.from_numpy(np.array(data_indexed)).to(device)
    data = data.unsqueeze(0) # [1, seq_length]

    config = AlbertConfig(hidden_size=768)
    model = ALBertClassifyModel(config, num_class=2, fc_dropout=DROPOUT)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():

        predict = model(data)
        predict_softmax = softmax(predict)
        print(predict_softmax)
        predict = torch.argmax(predict_softmax, dim=1)
        print(predict)


if __name__ == '__main__':
    # train()
    prediction()