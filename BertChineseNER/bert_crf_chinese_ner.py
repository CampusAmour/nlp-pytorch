#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: CampusAmour

import torch
import torch.nn as nn
import torchsnooper
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
SAVE_MODEL_PATH = './model/bert_chinese_crf_ner_model.pth'
MAX_SEQ_LENGTH = 128
DROPOUT = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_FILE_PATH = './data/example.train'
EVAL_FILE_PATH = './data/example.dev'
TEST_FILE_PATH = './data/example.test'
tokenizer = BertTokenizer.from_pretrained(VOCAB_PATH)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_index = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, START_TAG: 7, STOP_TAG: 8}


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
        for i in range(batch_size):
            targets_batch[i] = np.array(targets_batch[i] + [tag_to_index['O']] * (len(targets_batch[0])-len(targets_batch[i])))
        targets_batch = torch.from_numpy(np.array(targets_batch)).to(device)
        targets_length_batch = torch.from_numpy(np.array(targets_length[begin: end])).to(device)

        yield sentences_batch, targets_batch, targets_length_batch


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BertCRFNERModel(nn.Module):
    def __init__(self, config, tag_to_index, fc_dropout=0.1):
        super(BertCRFNERModel, self).__init__()
        # self.bert = BertModel(config)
        self.tag_to_index = tag_to_index
        self.hidden_to_tag = len(tag_to_index)
        self.bert = BertModel.from_pretrained(MODEL_PATH)

        # 锁梯度,做迁移学习
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.fc = nn.Linear(config.hidden_size, self.hidden_to_tag)
        self.dropout = nn.Dropout(fc_dropout)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.hidden_to_tag, self.hidden_to_tag))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_index[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_index[STOP_TAG]] = -10000


    def _get_bert_features(self, sentences):
        bert_hidden, _ = self.bert(sentences)
        bert_feats = self.fc(bert_hidden)
        return bert_feats


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.hidden_to_tag), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_index[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.hidden_to_tag):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.hidden_to_tag)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha


    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_index[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_index[STOP_TAG], tags[-1]]
        return score

    # @torchsnooper.snoop()
    def neg_log_likelihood(self, sentences, targets, targets_length):
        features = self._get_bert_features(sentences)
        scores = torch.zeros(1).to(device)
        for idx, feats in enumerate(features):
            forward_score = self._forward_alg(feats[1: targets_length[idx]+1, :])
            gold_score = self._score_sentence(feats[1: targets_length[idx]+1, :], targets[idx, :targets_length[idx]])
            scores += (forward_score - gold_score)
        return scores


    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.hidden_to_tag), -10000.).to(device)
        init_vvars[0][self.tag_to_index[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.hidden_to_tag):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_index[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_index[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    # for test
    def forward(self, sentences, targets_length):
        # Get the emission scores from the Bert
        features = self._get_bert_features(sentences)
        targets_sequences = []
        for idx, feats in enumerate(features):
            # Find the best path, given the features.
            _, tag_seq = self._viterbi_decode(feats[1: targets_length[idx]+1, :])
            targets_sequences.extend(tag_seq)
        return targets_sequences


def evaluate(model, eval_sentences, eval_targets, eval_targets_length, batch_size):
    total_loss, total_count = 0., 0
    batch_num = len(eval_sentences) // BATCH_SIZE
    batch = generate_batch(eval_sentences, eval_targets, eval_targets_length, batch_size)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            eval_sentences_batch, eval_targets_batch, eval_targets_length_batch = next(batch)
            loss = model.neg_log_likelihood(eval_sentences_batch, eval_targets_batch,
                                            eval_targets_length_batch)
            total_loss += loss.cpu().item()
    return total_loss / (batch_num * BATCH_SIZE)


def test(model, test_sentences, test_targets, test_targets_length, batch_size):
    batch_num = len(test_sentences) // BATCH_SIZE
    batch = generate_batch(test_sentences, test_targets, test_targets_length, batch_size)
    model.eval()
    total_predicts, total_targets = [], []
    with torch.no_grad():
        for i in range(batch_num):
            test_sentences_batch, test_targets_batch, test_targets_length_batch = next(batch)
            predicts = model(test_sentences_batch, test_targets_length_batch)

            total_predicts.extend(predicts)

            for idx, targets in enumerate(test_targets_batch):
                total_targets.extend(list(targets[:test_targets_length_batch[idx]].cpu().numpy()))

    print(total_targets)
    print(total_predicts)

    return f1_score(np.array(total_targets), np.array(total_predicts), average='macro'),\
           accuracy_score(np.array(total_targets), np.array(total_predicts))


def train():
    config = BertConfig()
    model = BertCRFNERModel(config, tag_to_index)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_sentences, train_targets, train_targets_length = load_data(TRAIN_FILE_PATH)
    eval_sentences, eval_targets, eval_targets_length = load_data(EVAL_FILE_PATH)
    test_sentences, test_targets, test_targets_length = load_data(TEST_FILE_PATH)

    print(len(train_sentences))

    start = time.time()
    try:
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.
            batch_num = len(train_sentences) // BATCH_SIZE
            batch = generate_batch(train_sentences, train_targets, train_targets_length, BATCH_SIZE)
            for i in tqdm(range(batch_num)):
                train_sentences_batch, train_targets_batch, train_targets_length_batch = next(batch) # labels_batch: [batch_size]
                optimizer.zero_grad()
                loss = model.neg_log_likelihood(train_sentences_batch, train_targets_batch, train_targets_length_batch)
                total_loss += loss.cpu().item()
                loss.backward()
                optimizer.step()

            eval_loss = evaluate(model, eval_sentences, eval_targets, eval_targets_length, BATCH_SIZE)
            f1_score_test, accuracy_test = test(model, test_sentences, test_targets, test_targets_length, BATCH_SIZE)

            print('epoch %d, loss_train %.4f, loss_eval %.4f, f1_score_test %.4f, accuracy_test %.4f, time %.2fmin' %\
                  (epoch+1, total_loss/(batch_num * BATCH_SIZE), eval_loss, f1_score_test, accuracy_test, (time.time()-start)/60))
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
    except KeyboardInterrupt:
        print('检测到外部中断,训练结束,模型已自动保存~')
        path = './model/epoch_' + str(epoch) + '_bert_chinese_crf_ner_model.pth'
        torch.save(model.state_dict(), path)


train()