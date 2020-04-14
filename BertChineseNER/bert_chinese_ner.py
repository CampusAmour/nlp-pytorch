#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.metrics import f1_score, accuracy_score

MODEL_PATH = './bert-base-chinese'
VOCAB_PATH = MODEL_PATH + '/bert-base-chinese-vocab.txt'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-5
SAVE_MODEL_PATH = './model/bert_chinese_ner_model.pth'
MAX_SEQ_LENGTH = 128
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_FILE_PATH = './data/example.train'
EVAL_FILE_PATH = './data/example.dev'
TEST_FILE_PATH = './data/example.test'
tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)

tag_to_index = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}


# def load_data(file_path, max_seq_length=MAX_SEQ_LENGTH):
#     sentences, targets, targets_length = [], [], []
#     sentence, target = [], []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.rstrip('\n')
#             if line == '':
#                 if len(sentence) <= max_seq_length:
#                     sentence.insert(0, "[CLS]")
#                     sentence.append("[SEP]")
#                     sentence_indexed = tokenizer.convert_tokens_to_ids(sentence)
#                     data_pad = [tokenizer.pad_token_id] * (max_seq_length + 2)
#                     seq_length = len(sentence_indexed)
#                     data_pad[:seq_length] = sentence_indexed[:seq_length]
#
#                     target = [ner_to_index[item] for item in target]
#                     sentences.append(np.array(data_pad))
#                     targets.append(np.array(target))
#                     targets_length.append(len(target))
#
#                 sentence, target = [], []
#             else:
#                 word, label = line.split(' ')
#                 sentence.append(word)
#                 target.append(label)
#
#     return sentences, targets, targets_length


def load_data(file_path, max_seq_length=MAX_SEQ_LENGTH):
    total_data, data = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '':
                total_data.append(data)
                data = []
            else:
                data.append(line)
    total_data = sorted(total_data, key=lambda data: len(data), reverse=True)
    for i in range(len(total_data)):
        if len(total_data[i]) <= max_seq_length:
            break
    total_data = total_data[i:]
    sentences, targets, targets_length = [], [], []
    sentence, target = [], []
    for data in total_data:
        for item in data:
            word, label = item.split(' ')
            sentence.append(word)
            target.append(label)
        sentence.insert(0, "[CLS]")
        sentence.append("[SEP]")
        sentence_indexed = tokenizer.convert_tokens_to_ids(sentence)
        data_pad = [tokenizer.pad_token_id] * (max_seq_length + 2)
        seq_length = len(sentence_indexed)
        data_pad[:seq_length] = sentence_indexed[:seq_length]

        target = [tag_to_index[num] for num in target]
        sentences.append(np.array(data_pad))
        targets.append(target)
        targets_length.append(len(target))
        sentence, target = [], []
    return sentences, targets, targets_length


def generate_batch(sentences, targets, targets_length, batch_size=64):
    batch_num = len(sentences) // batch_size
    for k in range(batch_num):
        begin = k * batch_size
        end = begin + batch_size
        sentences_batch = torch.from_numpy(np.array(sentences[begin: end])).to(device)
        targets_batch = list(targets[begin: end])
        target_words_batch = []
        for target_seq in targets_batch:
            target_words_batch.extend(target_seq)
        target_words_batch = torch.from_numpy(np.array(target_words_batch)).to(device)
        targets_length_batch = torch.from_numpy(np.array(targets_length[begin: end])).to(device)

        yield sentences_batch, target_words_batch, targets_length_batch


class BertNERModel(nn.Module):
    def __init__(self, config, num_class=7, fc_dropout=0.1):
        super(BertNERModel, self).__init__()
        # self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained(MODEL_PATH)
        self.fc = nn.Linear(config.hidden_size, num_class)
        self.dropout = nn.Dropout(fc_dropout)

    def forward(self, x, max_target_length):
        # bert return: (last_hidden_​​state, pooler_output)
        hidden_states, _ = self.bert(x)
        return self.dropout(self.fc(hidden_states[:, 1: max_target_length+1, :]))


def evaluate(model, eval_sentences, eval_targets, eval_targets_length, criterion, batch_size):
    total_loss, total_count = 0., 0
    batch_num = len(eval_sentences) // BATCH_SIZE
    batch = generate_batch(eval_sentences, eval_targets, eval_targets_length, batch_size)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            eval_sentences_batch, eval_target_words_batch, eval_targets_length_batch = next(batch)

            outputs = model(eval_sentences_batch, eval_targets_length_batch[0])
            new_outputs = torch.zeros([1, len(tag_to_index)]).to(device)

            for idx, output in enumerate(outputs):
                new_outputs = torch.cat((new_outputs, output[: eval_targets_length_batch[idx], :]), dim=0)
            new_outputs = new_outputs[1:, :]

            loss = criterion(new_outputs, eval_target_words_batch)
            total_loss += (loss.cpu().item() * eval_targets_length_batch.sum().cpu().item())
            total_count += eval_targets_length_batch.sum().cpu().item()
    return total_loss / total_count


def test(model, test_sentences, test_targets, test_targets_length, batch_size):
    batch_num = len(test_sentences) // BATCH_SIZE
    batch = generate_batch(test_sentences, test_targets, test_targets_length, batch_size)
    model.eval()
    softmax = nn.Softmax(dim=1)
    total_predicts, total_targets = [], []
    with torch.no_grad():
        for i in range(batch_num):
            test_sentences_batch, test_target_words_batch, test_targets_length_batch = next(batch)

            predicts = model(test_sentences_batch, test_targets_length_batch[0])
            new_predicts = torch.zeros([1, len(tag_to_index)]).to(device)

            for idx, predict in enumerate(predicts):
                new_predicts = torch.cat((new_predicts, predict[: test_targets_length_batch[idx], :]), dim=0)
            new_predicts = new_predicts[1:, :]
            new_predicts = torch.argmax(softmax(new_predicts), dim=1)

            total_predicts.extend(list(new_predicts.cpu().numpy()))
            total_targets.extend(list(test_target_words_batch.cpu().numpy()))

    return f1_score(np.array(total_targets), np.array(total_predicts), average='macro'),\
           accuracy_score(np.array(total_targets), np.array(total_predicts))


def train():
    config = BertConfig()
    model = BertNERModel(config)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_sentences, train_targets, train_targets_length = load_data(TRAIN_FILE_PATH)
    eval_sentences, eval_targets, eval_targets_length = load_data(EVAL_FILE_PATH)
    test_sentences, test_targets, test_targets_length = load_data(TEST_FILE_PATH)

    start = time.time()
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss, total_count = 0., 0
            batch_num = len(train_sentences) // BATCH_SIZE
            batch = generate_batch(train_sentences, train_targets, train_targets_length, BATCH_SIZE)
            for i in tqdm(range(batch_num)):
                train_sentences_batch, train_target_words_batch, train_targets_length_batch = next(batch) # labels_batch: [batch_size]

                outputs = model(train_sentences_batch, train_targets_length_batch[0])
                new_outputs = torch.zeros([1, len(tag_to_index)]).to(device)
                for idx, output in enumerate(outputs):
                    new_outputs = torch.cat((new_outputs, output[: train_targets_length_batch[idx], :]), dim=0)
                new_outputs = new_outputs[1:, :]
                # print(new_outputs.shape) # [batch_total_words, class]

                optimizer.zero_grad()
                loss = criterion(new_outputs, train_target_words_batch)
                total_loss += (loss.cpu().item() * train_targets_length_batch.sum().cpu().item())
                total_count += train_targets_length_batch.sum().cpu().item()
                loss.backward()
                optimizer.step()

            eval_loss = evaluate(model, eval_sentences, eval_targets, eval_targets_length, criterion, BATCH_SIZE)
            f1_score_test, accuracy_test = test(model, test_sentences, test_targets, test_targets_length, BATCH_SIZE)

            print('epoch %d, loss_train %.4f, loss_eval %.4f, f1_score_test %.4f, accuracy_test %.4f, time %.2fmin' %\
                  (epoch+1, total_loss/total_count, eval_loss, f1_score_test, accuracy_test, (time.time()-start)/60))
    except KeyboardInterrupt:
        print('检测到外部中断,训练结束,模型已自动保存~')
        path = './model/epoch_' + str(epoch) + '_bert_chinese_ner_model.pth'
        torch.save(model.state_dict(), path)


train()