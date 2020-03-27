import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from data_process import en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words,\
    encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, \
    encoder_inputs_test, decoder_inputs_test, decoder_targets_test,\
    get_batch, generate_predict_sentence


USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
EMBED_DIM = 128
ENCODER_HIDDEN_DIM = DECODER_HIDDEN_DIM = 256
LEARNING_RATE = 1e-4
EPOCHS = 512
BATCH_SIZE = 64
MODEL_PATH = './seq2seq_model/seq2seq_attention_model.pth'

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_hidden_dim, decoder_hidden_dim, pad_idx, drop_out=0.2):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, encoder_hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(2*encoder_hidden_dim, decoder_hidden_dim)

    def forward(self, x, x_lengths):
        # x已经按照长度排序
        embedded = self.dropout(self.embed(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded.permute(1, 0, 2), x_lengths)
        packed_outputs, hidden_state = self.gru(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # hidden_state: [num_layers*num_direction, batch_size, hidden_dim]
        # hidden_state: [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer_1, ..., forward_layer_n, backward_layer_n]
        # 将最后hidden_state最后两个forward_layer_n和backward_layer_n拼接在一起后[batch_size, hidden_dim*2]做一个线性变换，再接tanh激活
        hidden_state = torch.tanh(self.fc(torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)))
        return outputs, hidden_state


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention, self).__init__()
        # encoder_dim to decoder_dim
        self.linear_encoder = nn.Linear(2*encoder_hidden_dim, decoder_hidden_dim, bias=False)
        self.linear_decoder = nn.Linear(2*encoder_hidden_dim, decoder_hidden_dim)

    def forward(self, decoder_input, encoder_outputs, encoder_hidden_state):
        # encoder_outputs: [encoder_seq_lengths, batch_size, encoder_hidden_dim]
        # 先将encoder_outputs.permute,再经过一次linear_encoder,因为encoder_hidden_dim = 2 * decoder_hidden_dim
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_hidden_state: [batch_size, decoder_hidden_dim], squeeze成 [batch_size, decoder_hidden_dim, 1]
        attention = torch.bmm(self.linear_encoder(encoder_outputs), encoder_hidden_state.unsqueeze(2))
        # attention: [batch, seq_length, 1]
        attention = F.softmax(attention, dim=1) # 按行(seq_length)进行softmax
        # attention.permute(0, 2, 1): [batch, 1, seq_length]
        context = torch.bmm(attention.permute(0, 2, 1), encoder_outputs).squeeze(1)
        # context: [batch_size, 1, decoder_hidden_dim]
        return self.linear_decoder(context)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_hidden_dim, decoder_hidden_dim,  pad_idx, drop_out=0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
        self.gru = nn.GRU(embed_dim+decoder_hidden_dim, decoder_hidden_dim, )
        self.fc = nn.Linear(decoder_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, y, encoder_outputs, encoder_hidden_state):
        embedded = self.dropout(self.embed(y))
        # embedded = embedded.permute(1, 0, 2) # [seq_lengths, batch_size, hidden_dim]
        # print(embedded.shape)
        # print(encoder_hidden_state.repeat(embedded.shape[0], 1, 1).shape)
        # encoder_embedded = torch.cat((embedded, encoder_hidden_state.repeat(embedded.shape[0], 1, 1)), 2)

        context = self.attention(decoder_input=y, encoder_outputs=encoder_outputs, encoder_hidden_state=encoder_hidden_state)

        encoder_embedded = torch.cat((embedded, context), 1)
        # print(encoder_embedded.shape)
        # 将embedded与encoder_hidden_state拼起来 # [seq_lengths, batch_size, embed_dim+hidden_dim]

        outputs, hidden_state = self.gru(encoder_embedded.unsqueeze(0), encoder_hidden_state.unsqueeze(0))
        # return self.softmax(self.fc(outputs[-1,:,:]))
        return self.fc(outputs[0]), hidden_state.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, x_lengths, y, y_lengths):
        encoder_outputs, encoder_hidden_state = self.encoder(x, x_lengths)
        outputs = torch.zeros(y.shape[1], y.shape[0], len(ch_word_to_index)).to(device)
        for i in range(y.shape[1]):
            # y[:, i]: [64]
            # encoder_hidden_state: [64, 256]
            decoder_output, encoder_hidden_state = self.decoder(y[:,i], encoder_outputs, encoder_hidden_state)
            outputs[i] = decoder_output
        # outputs = self.decoder(y, y_lengths, encoder_hidden_state)
        # print(outputs.shape)
        return outputs

    def translate(self, x, x_length, max_length=50):
        encoder_outputs, encoder_hidden_state = self.encoder(x, x_length)
        predict = []
        word = torch.Tensor([ch_word_to_index['<GO>']]).to(device).long()
        for i in range(max_length):
            predict_output, encoder_hidden_state = self.decoder(word, encoder_outputs, encoder_hidden_state)
            word = torch.argmax(predict_output, dim=1)
            if word.cpu().item() == ch_word_to_index['<EOS>']:
                predict.append(word.cpu().item())
                break
            predict.append(word.cpu().item())
        return predict


def calculate_bleu(references, candidates):
    # reference: 3维
    # candidate: 2维
    # reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
    # candidate = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    score = 0.
    smooth = SmoothingFunction()  # 定义平滑函数对象
    length = len(candidates)

    for i in range(length):
        score += sentence_bleu([references[i]], candidates[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method2)
    return score / length


def evaluate(model, encoder_inputs_test, decoder_inputs_test, decoder_targets_test, criterion, batch_size=BATCH_SIZE):
    candidates = []
    batch_num = len(encoder_inputs_train) // BATCH_SIZE
    batch = get_batch(encoder_inputs_test, decoder_inputs_test, decoder_targets_test, BATCH_SIZE)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            encoder_inputs_batch, _, decoder_targets_batch, encoder_batch_length, decoder_batch_length = next(batch)
            encoder_inputs_batch = torch.from_numpy(encoder_inputs_batch).to(device).long()
            encoder_batch_length = torch.from_numpy(encoder_batch_length).to(device).long()
            encoder_inputs_batch = encoder_inputs_batch.unsqueeze(0)
            encoder_batch_length = encoder_batch_length.unsqueeze(0)
            for j in range(batch_size):
                predict = model.translate(encoder_inputs_batch[:, 1, :], encoder_batch_length[:, 1])
                candidates.append(predict)
    return calculate_bleu(decoder_targets_test, candidates)


'''
def evaluate(model, encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, criterion, batch_size=BATCH_SIZE):
    total_loss = 0.
    batch_num = len(encoder_inputs_train) // BATCH_SIZE
    batch = get_batch(encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, BATCH_SIZE)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            encoder_inputs_batch, decoder_inputs_batch, decoder_targets_batch, encoder_batch_length, decoder_batch_length = next(
                batch)
            encoder_inputs_batch = torch.from_numpy(encoder_inputs_batch).to(device).long()
            encoder_batch_length = torch.from_numpy(encoder_batch_length).to(device).long()

            decoder_inputs_batch = torch.from_numpy(decoder_inputs_batch).to(device).long()
            decoder_targets_batch = torch.from_numpy(decoder_targets_batch).to(device).long()
            decoder_batch_length = torch.from_numpy(decoder_batch_length).to(device).long()

            outputs = model(encoder_inputs_batch, encoder_batch_length, decoder_inputs_batch, decoder_batch_length)
            outputs = outputs.view(-1, outputs.shape[-1])
            decoder_targets_batch = decoder_targets_batch.view(-1)
            # outputs = torch.Tensor(outputs)
            loss = criterion(outputs, decoder_targets_batch)
            total_loss += loss.cpu().item()
    return total_loss / batch_num
'''


def train():
    encoder = Encoder(vocab_size=en_total_words, embed_dim=EMBED_DIM, encoder_hidden_dim=ENCODER_HIDDEN_DIM,\
                      decoder_hidden_dim=DECODER_HIDDEN_DIM, pad_idx=en_word_to_index['<PAD>'])
    decoder = Decoder(vocab_size=ch_total_words, embed_dim=EMBED_DIM, encoder_hidden_dim=ENCODER_HIDDEN_DIM,\
                      decoder_hidden_dim=DECODER_HIDDEN_DIM, pad_idx=ch_word_to_index['<PAD>'])
    model = Seq2Seq(encoder, decoder)
    model.to(device)
    # print(model)
    criterion = nn.CrossEntropyLoss(ignore_index=ch_word_to_index['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.
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
        if (epoch + 1) % 20 == 0:
            acc = evaluate(model, encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, criterion, batch_size=BATCH_SIZE)
            print('epoch %d, loss %.4f, accuracy %.6f' % (epoch + 1, total_loss / batch_num, acc))
        else:
            print('epoch %d, loss %.4f' % (epoch + 1, total_loss / batch_num))
    torch.save(model.state_dict(), MODEL_PATH)


def test(english):
    encoder_inputs, encoder_inputs_length = generate_predict_sentence(english, en_word_to_index)
    encoder_inputs = torch.from_numpy(encoder_inputs).to(device).long()
    encoder_inputs_length = torch.from_numpy(encoder_inputs_length).to(device).long()
    encoder = Encoder(vocab_size=en_total_words, embed_dim=EMBED_DIM, encoder_hidden_dim=ENCODER_HIDDEN_DIM,\
                      decoder_hidden_dim=DECODER_HIDDEN_DIM, pad_idx=en_word_to_index['<PAD>'])
    decoder = Decoder(vocab_size=ch_total_words, embed_dim=EMBED_DIM, encoder_hidden_dim=ENCODER_HIDDEN_DIM,\
                      decoder_hidden_dim=DECODER_HIDDEN_DIM, pad_idx=ch_word_to_index['<PAD>'])
    model = Seq2Seq(encoder, decoder)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    predict = model.translate(encoder_inputs, encoder_inputs_length)
    print(predict)

    predict = [ch_index_to_word[item] for item in predict]
    print(predict)


train()
# english = 'I come from Chongqing!'
# test(english)