import torch
import torch.nn as nn
# from data_process import en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words, \
#     encoder_inputs, decoder_inputs, decoder_targets, get_batch, generate_predict_sentence

from data_process import en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words,\
    encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, \
    encoder_inputs_test, decoder_inputs_test, decoder_targets_test,\
    get_batch, generate_predict_sentence


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
EMBED_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
EPOCHS = 512
BATCH_SIZE = 64
MODEL_PATH = './seq2seq_model/seq2seq_model.pth'

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, drop_out=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, x_lengths):
        # x已经按照长度排序
        embedded = self.dropout(self.embed(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded.permute(1, 0, 2), x_lengths)
        packed_outputs, hidden_state = self.gru(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        return outputs[-1,:,:]


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, drop_out=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim+hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, y, encoder_hidden_state):
        embedded = self.dropout(self.embed(y))
        # embedded = embedded.permute(1, 0, 2) # [seq_lengths, batch_size, hidden_dim]
        # print(embedded.shape)
        # print(encoder_hidden_state.repeat(embedded.shape[0], 1, 1).shape)
        # encoder_embedded = torch.cat((embedded, encoder_hidden_state.repeat(embedded.shape[0], 1, 1)), 2)

        encoder_embedded = torch.cat((embedded, encoder_hidden_state), 1)
        # print(encoder_embedded.shape)
        # 将embedded与encoder_hidden_state拼起来 # [seq_lengths, batch_size, embed_dim+hidden_dim]

        outputs, hidden_state = self.gru(encoder_embedded.unsqueeze(0), encoder_hidden_state.unsqueeze(0))
        # return self.softmax(self.fc(outputs[-1,:,:]))
        return self.fc(outputs[0])



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_hidden_state = self.encoder(x, x_lengths)
        outputs = torch.zeros(y.shape[1], y.shape[0], len(ch_word_to_index)).to(device)
        for i in range(y.shape[1]):
            # y[:, i]: [64]
            # encoder_hidden_state: [64, 256]
            decoder_output = self.decoder(y[:,i], encoder_hidden_state)
            outputs[i] = decoder_output
        # outputs = self.decoder(y, y_lengths, encoder_hidden_state)
        # print(outputs.shape)
        return outputs

    def translate(self, x, x_length, max_length=50):
        encoder_hidden_state = self.encoder(x, x_length)
        predict = []
        word = torch.Tensor([ch_word_to_index['<GO>']]).to(device).long()
        for i in range(max_length):
            predict_output = self.decoder(word, encoder_hidden_state)
            word = torch.argmax(predict_output, dim=1)
            if word.cpu().item() == ch_word_to_index['<EOS>']:
                break
            predict.append(word.cpu().item())
        return predict



def train():
    encoder = Encoder(vocab_size=en_total_words, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, pad_idx=en_word_to_index['<PAD>'])
    decoder = Decoder(vocab_size=ch_total_words, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, pad_idx=ch_word_to_index['<PAD>'])
    model = Seq2Seq(encoder, decoder)
    model.to(device)
    # print(model)
    criterion = nn.CrossEntropyLoss(ignore_index=ch_word_to_index['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = total_num_words = 0.
        batch_num = len(encoder_inputs_train) // BATCH_SIZE
        batch = get_batch(encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, BATCH_SIZE)
        for i in range(batch_num):
            encoder_inputs_batch, decoder_inputs_batch, decoder_targets_batch, encoder_batch_length, decoder_batch_length = next(batch)
            encoder_inputs_batch = torch.from_numpy(encoder_inputs_batch).to(device).long()
            encoder_batch_length = torch.from_numpy(encoder_batch_length).to(device).long()

            decoder_inputs_batch = torch.from_numpy(decoder_inputs_batch).to(device).long()
            decoder_targets_batch = torch.from_numpy(decoder_targets_batch).to(device).long()
            decoder_batch_length = torch.from_numpy(decoder_batch_length).to(device).long()
            outputs = model(encoder_inputs_batch, encoder_batch_length, decoder_inputs_batch, decoder_batch_length)
            # print(outputs.shape)
            # print(decoder_targets_batch.shape)

            outputs = outputs.view(-1, outputs.shape[-1]) # [batch_size*seq_length, vocab_size]
            # print(decoder_targets_batch.view(-1).size())
            decoder_targets_batch = decoder_targets_batch.view(-1) # [batch_size*seq_length]
            # outputs = torch.Tensor(outputs)
            loss = criterion(outputs, decoder_targets_batch)
            total_loss += loss.cpu().item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        print('epoch %d, loss %.4f' % (epoch + 1, total_loss / batch_num))
    torch.save(model.state_dict(), MODEL_PATH)


def test(english):
    encoder_inputs, encoder_inputs_length = generate_predict_sentence(english, en_word_to_index)
    encoder_inputs = torch.from_numpy(encoder_inputs).to(device).long()
    encoder_inputs_length = torch.from_numpy(encoder_inputs_length).to(device).long()
    encoder = Encoder(vocab_size=en_total_words, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                      pad_idx=en_word_to_index['<PAD>'])
    decoder = Decoder(vocab_size=ch_total_words, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                      pad_idx=ch_word_to_index['<PAD>'])
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    predict = model.translate(encoder_inputs, encoder_inputs_length)
    # print(predict)

    predict = [ch_index_to_word[item] for item in predict]
    print(predict)


train()
english = 'I come from Chongqing.'
test(english)