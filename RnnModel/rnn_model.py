#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import torch.nn as nn
from data_process import get_data

SEED = 7
split_ratio = 0.8
SEQ_LENGTH = 256
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
VOCAB_SIZE = 10000
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2
learning_rate = 1e-4
NUM_EPOCHS = 8
MODEL_PATH = './models/bi_rnn_model.pth'

vocab, train_iterator, valid_iterator, test_iterator = get_data(SEQ_LENGTH, SEED, split_ratio, VOCAB_SIZE-2, BATCH_SIZE, device)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, bidirectional=False):
        super(RNNModel, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, bidirectional=bidirectional, dropout=0.1)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 4, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedding = self.embed(text)
        output, hidden_state = self.rnn(embedding)
        output = output.permute(1, 0, 2)
        # 获取 time seq 维度中的最后一个向量
        if self.bidirectional:
            output = torch.cat([output[:, 0, :], output[:, -1, :]], dim=1)
        else:
            output = output[:, -1, :] # 把每个batch中序列的最后一维输出给取出来
        return self.fc(output)

def evaluate(model, data):
    model.eval()
    acc_sum, total_count = 0.0, 0.0
    iterator = iter(data)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            inputs, labels = batch.text, batch.label
            outputs = model(inputs.to(device))
            acc_sum += (outputs.argmax(dim=1) == labels.to(device)).float().sum().item()
            total_count += inputs.shape[0]
    return acc_sum / total_count


def train(bidirectional=False):
    model = RNNModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, bidirectional)
    # print(model)
    loss_fn = nn.CrossEntropyLoss()
    if USE_CUDA:
        model = model.to(device)
        loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        model.train() # 是否启用drop_out
        iterator = iter(train_iterator)
        train_loss, batch_count = 0.0, 0
        for i, batch in enumerate(iterator):
            inputs, labels = batch.text, batch.label
            if USE_CUDA:
                inputs = inputs.to(device)
                labels = labels.to(device)

            # 防止梯度叠加
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.long())

            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()

            # print(loss.item())
            train_loss += loss.cpu().item()
            batch_count += 1
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch+1, train_loss / batch_count, evaluate(model, valid_iterator),
                                                                      evaluate(model, test_iterator)))
    torch.save(model.state_dict(), MODEL_PATH)

def predict_sentimen(vocab, sentence, bidirectional=False):
    model = RNNModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, bidirectional)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    sentence = sentence.split()
    sentence = [vocab.stoi.get(word, vocab.stoi['<unk>']) for word in sentence]
    if len(sentence) < SEQ_LENGTH:
        sentence += [vocab.stoi['<pad>']] * (SEQ_LENGTH - len(sentence))
    # print(len(sentence))
    sentence = torch.LongTensor(sentence).view(-1, 1).to(device)
    label = model(sentence).argmax(dim=1).cpu().item()
    return 'positive' if label == 1 else 'negativate'

train(bidirectional=True)

print(predict_sentimen(vocab, sentence='while this movie is so good', bidirectional=True))