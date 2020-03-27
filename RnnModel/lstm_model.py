import torch
import torch.nn as nn
import numpy as np
from data_process import get_data

SEED = 7
split_ratio = 0.8
SEQ_LENGTH = 512
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
VOCAB_SIZE = 20000
EMBED_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 2
NUM_LAYERS = 4
learning_rate = 3e-4
NUM_EPOCHS = 64
MODEL_PATH = './models/bi_lstm_model.pth'
BIDIRECITONAL = True
DROP_OUT = 0.4

vocab, train_iterator, valid_iterator, test_iterator = get_data(SEQ_LENGTH, SEED, split_ratio, VOCAB_SIZE-2, BATCH_SIZE, device)

PAD_IDX = vocab.stoi['<pad>']

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, drop_out, pad_idx):
        super(LSTMModel, self).__init__()
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx) #　padding_idx: 输出遇到此下标时用零填充
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=drop_out)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(drop_out)


    def forward(self, text, text_lengths):
        # text: [seq_length, batch_size]
        # embedden: [seq_length, batch_size, embed_dim]
        embedded = self.dropout(self.embed(text))
        print(embedded.shape)
        print(text_lengths)
        a = input()
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden_state, cell) = self.lstm(packed_embedded)

        # unpack sequence

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output: [seq_length, batch_size, hidden_dim * num_directions]
        # hidden = [num_layers * num_directions, batch_size, hidden_dim]
        # cell = [num_layers * num_directions, batch_size, hidden_dim]
        hidden_state = self.dropout(torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim=1)) # [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer 1, ..., forward_layer_n, backward_layer n]

        return self.fc(hidden_state)


def count_parameters():
    model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECITONAL, DROP_OUT, PAD_IDX)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

# count_parameters()


def evaluate(model, data):
    model.eval()
    acc_sum, total_count = 0.0, 0.0
    iterator = iter(data)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            (inputs, text_lengths), labels = batch.text, batch.label
            outputs = model(inputs.to(device), text_lengths)
            acc_sum += (outputs.argmax(dim=1) == labels.to(device)).float().sum().item()
            total_count += inputs.shape[0]
    return acc_sum / total_count


def train():
    model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECITONAL, DROP_OUT, PAD_IDX)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        model = model.to(device)
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        model.train() # 是否启用drop_out
        iterator = iter(train_iterator)
        train_loss, batch_count = 0.0, 0
        for i, batch in enumerate(iterator):
            (inputs, text_lengths), labels = batch.text, batch.label
            # print(text_lengths)
            if USE_CUDA:
                inputs = inputs.to(device)
                labels = labels.to(device)

            # 防止梯度叠加
            optimizer.zero_grad()

            # forward
            outputs = model(inputs, text_lengths)
            loss = criterion(outputs, labels.long())

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





def predict_sentimen(vocab, sentence):
    model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, BIDIRECITONAL, DROP_OUT, PAD_IDX)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    sentence = sentence.split()
    sentence = [vocab.stoi.get(word, vocab.stoi['<unk>']) for word in sentence]
    sentence_length = np.array([len(sentence)])
    sentence_length = torch.from_numpy(sentence_length).long()

    sentence = torch.LongTensor(sentence).view(-1, 1).to(device)
    outputs = model(sentence, sentence_length)
    print(outputs)
    label = outputs.argmax(dim=1).cpu().item()
    # label = model(sentence, sentence_length).argmax(dim=1).cpu().item()
    return 'positive' if label == 1 else 'negativate'

train()

# print(predict_sentimen(vocab, sentence='This film is great'))
