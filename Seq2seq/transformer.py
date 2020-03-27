import math
import torch
import torch.nn as nn
import numpy as np
import time
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from data_process import en_word_to_index, en_index_to_word, en_total_words, ch_word_to_index, ch_index_to_word, ch_total_words,\
    encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, \
    encoder_inputs_test, decoder_inputs_test, decoder_targets_test,\
    get_batch, generate_predict_sentence

USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

BATCH_SIZE = 32
LEARNING_RATE = 3e-5
D_MODEL = 512
EPOCHS = 300
MAX_SEQ_LENGTH = 40 # 以English最长序列长度为准
MODEL_PATH = './seq2seq_model/transformer_model.pth'


def sequence_padding(sequences, batch_length, max_seq_lenth=40):
    sequences = torch.from_numpy(sequences).long()
    sequences_padding = torch.zeros(sequences.shape[0], max_seq_lenth).long()
    sequence_length = max(batch_length)
    for index, sequence in enumerate(sequences):
        sequences_padding[index, : sequence_length] = sequence
    return sequences_padding.to(device)


def padding_mask(sequences):
    batch_size, sequence_length = sequences.shape
    # `<PAD>` is 0
    pad_mask = sequences.eq(0).unsqueeze(1) # pad_mask: [batch_size, 1, sequence_length]
    pad_mask = pad_mask.expand(-1, sequence_length, -1)  # pad_mask: [batch_size, sequence_length, sequence_length]
    return pad_mask


def sequence_mask(sequences):
    # sequences: [batch_size, max_seq_length]
    mask = torch.triu(torch.ones(sequences.shape[-1], sequences.shape[-1]), diagonal=1).to(device).long()
    # sequences = sequences.unsqueeze(1).expand(-1, sequences.shape[-1], -1)
    # mask_sequences = torch.mul(sequences, mask) # mask_sequences: [batch_size, max_seq_length, max_seq_length]
    # return mask_sequences
    return mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.2):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attention_mask=None):
        # q, k, v: [batch_size, max_seq_length, hidden_dim]
        # scale=1/k_hidden_dim
        # attention_mask: [batch_size, max_seq_length, max_seq_length]
        scale = 1. / math.sqrt(k.shape[-1])
        attention = torch.bmm(q, k.permute(0, 2, 1)) * scale # q*k.t: [batch_size, seq_length, seq_length]
        if attention_mask is not None:
            attention = attention.masked_fill(attention_mask, -1e9)
        attention = self.dropout(self.softmax(attention))
        attention = torch.bmm(attention, v)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, per_head_hidden_dim=64, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.per_head_hidden_dim = per_head_hidden_dim
        self.linear_q = nn.Linear(d_model, num_heads*per_head_hidden_dim, bias=False)
        self.linear_k = nn.Linear(d_model, num_heads*per_head_hidden_dim, bias=False)
        self.linear_v = nn.Linear(d_model, num_heads*per_head_hidden_dim, bias=False)

        self.sacled_dot_product_attention = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(num_heads*per_head_hidden_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query_inputs, key_inputs, value_inputs, attention_mask=None):
        batch_size = query_inputs.shape[0]
        query = self.linear_q(query_inputs)
        key = self.linear_k(key_inputs)
        value = self.linear_v(value_inputs)

        # split by heads
        query = query.view(batch_size*self.num_heads, -1, self.per_head_hidden_dim) # [batch_size*num_heads, max_seq_lengths, per_head_hidden_dim]
        key = key.view(batch_size * self.num_heads, -1, self.per_head_hidden_dim)
        value = value.view(batch_size * self.num_heads, -1, self.per_head_hidden_dim)
        # print(attention_mask)

        if attention_mask is not None:
            attention_mask = attention_mask.repeat(self.num_heads, 1, 1) # attention_mask:[batch_size*num_heads, max_seq_lengths, max_seq_lengths]

        attention = self.sacled_dot_product_attention(query, key, value, attention_mask) # [batch_size*num_heads, max_seq_lengths, per_head_hidden_dim]
        attention = attention.view(batch_size, -1, self.num_heads*self.per_head_hidden_dim) # [batch_size, max_seq_lengths, num_heads*per_head_hidden_dim]

        outputs = self.fc(attention)
        outputs = self.layer_norm(query_inputs + outputs)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        # position_encoding: [max_seq_length, d_model]
        position_encoding = np.array([[pos / np.power(10000, 2.0 * i / d_model) for i in range(d_model)] for pos in range(max_seq_length)])

        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        self.position_encoding = torch.from_numpy(position_encoding)

    def forward(self, batch_size, batch_length):
        position_embedding = torch.zeros(batch_size, self.position_encoding.shape[0],self.position_encoding.shape[1])
        for index, (encoding, seq_length) in enumerate(zip(self.position_encoding, batch_length)):
            position_embedding[index, :seq_length, :] = self.position_encoding[:seq_length, :]
        return position_embedding.to(device).detach()


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, ffn_dim=2048, dropout=0.2):
        super(PositionalWiseFeedForward, self).__init__()
        # FFZ(X) = max(0, X*W1 + b1)W2 + b2
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, multi_attention_outputs):
        outputs = self.fc2(self.relu(self.fc1(multi_attention_outputs)))
        outputs = self.layer_norm(multi_attention_outputs + outputs)

        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, per_head_hidden_dim=64, ffn_dim=2048, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, per_head_hidden_dim, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, inputs, attention_mask=None):
        # multi-head self-attention
        attention = self.attention(inputs, inputs, inputs, attention_mask)
        # feed-forward network
        outputs = self.feed_forward(attention)
        return outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, pad_idx, d_model=512, num_layers=6, num_heads=8, per_head_hidden_dim=64, ffn_dim=2048, dropout=0.2):
        super(Encoder, self).__init__()
        # encoder stack
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, per_head_hidden_dim, ffn_dim, dropout) for _ in range(num_layers)])

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embed = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_length):
        embedded = self.dropout(self.embed(x))
        embedded += self.position_embed(x.shape[0], batch_length)

        self_attention_padding_mask = padding_mask(x)

        outputs = embedded
        for encoder in self.encoder_layers:
            outputs = encoder(outputs, self_attention_padding_mask)
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, per_head_hidden_dim=64, ffn_dim=2048, dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, per_head_hidden_dim, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, inputs, encoder_outputs, attention_mask=None, context_attention_mask=None):
        # masked multi-head self-attention
        masked_attention = self.attention(inputs, inputs, inputs, attention_mask)

        # multi-head self-attention
        attention = self.attention(encoder_outputs, encoder_outputs, masked_attention, context_attention_mask)

        # feed-forward network
        outputs = self.feed_forward(attention)
        return outputs


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, pad_idx, d_model=512, num_layers=6, num_heads=8, per_head_hidden_dim=64, ffn_dim=2048, dropout=0.2):
        super(Decoder, self).__init__()
        # decoder stack
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, per_head_hidden_dim, ffn_dim, dropout) for _ in range(num_layers)])

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embed = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, batch_length, encoder_outputs):
        embedded = self.dropout(self.embed(y))
        embedded += self.position_embed(y.shape[0], batch_length)
        self_attention_padding_mask = padding_mask(y)
        seq_mask = sequence_mask(y)

        self_attention_mask = torch.gt((self_attention_padding_mask+seq_mask), 0)

        outputs = embedded
        for decoder in self.decoder_layers:
            outputs = decoder(outputs, encoder_outputs, self_attention_padding_mask, self_attention_mask)
        return outputs


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, source_max_seq_length, target_vocab_size, target_max_seq_length, encoder_pad_idx, decoder_pad_idx, num_layers=6,\
                 d_model=512, num_heads=8, per_head_hidden_dim=64, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()
        self.encoder = Encoder(source_vocab_size, source_max_seq_length, encoder_pad_idx, d_model, num_layers, num_heads, per_head_hidden_dim, ffn_dim, dropout)
        self.decoder = Decoder(target_vocab_size, target_max_seq_length, decoder_pad_idx, d_model, num_layers, num_heads, per_head_hidden_dim, ffn_dim, dropout)
        self.linear = nn.Linear(d_model, target_vocab_size)
        self.softmax = nn.Softmax(dim=1) # for translate function

    def forward(self, encoder_input, encoder_length, decoder_input, decoder_length):
        encoder_outputs = self.encoder(encoder_input, encoder_length)

        decoder_outputs = self.decoder(decoder_input, decoder_length, encoder_outputs)
        outputs = self.linear(decoder_outputs)
        # return self.softmax(outputs)
        return outputs

    def translate(self, source, source_length):
        encoder_outputs = self.encoder(source, source_length)
        predict = [ch_word_to_index['<GO>']] + (MAX_SEQ_LENGTH-1) * [ch_word_to_index['<PAD>']]
        for i in range(MAX_SEQ_LENGTH-1):
            decoder_input = torch.Tensor(predict).unsqueeze(0).to(device).long()
            decoder_length = torch.Tensor([len(predict)]).to(device).long()
            decoder_outputs = self.decoder(decoder_input, decoder_length, encoder_outputs)
            outputs = self.linear(decoder_outputs)
            target_word = outputs[:, i, :]
            target_word = torch.argmax(self.softmax(target_word)).cpu().item()
            if target_word == ch_word_to_index['<EOS>']:
                break
            predict[i+1] = target_word
        return predict[1: i+1]


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


def evaluate(model, encoder_inputs_test, decoder_inputs_test, decoder_targets_test, criterion, batch_size):
    total_loss = 0.
    batch_num = len(encoder_inputs_test) // BATCH_SIZE
    batch = get_batch(encoder_inputs_test, decoder_inputs_test, decoder_targets_test, batch_size)
    model.eval()
    with torch.no_grad():
        for i in range(batch_num):
            encoder_inputs_batch, decoder_inputs_batch, decoder_targets_batch, encoder_batch_length, decoder_batch_length = next(batch)
            encoder_inputs_batch = sequence_padding(encoder_inputs_batch, encoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)
            decoder_inputs_batch = sequence_padding(decoder_inputs_batch, decoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)
            decoder_targets_batch = sequence_padding(decoder_targets_batch, decoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)

            outputs = model(encoder_inputs_batch, encoder_batch_length, decoder_inputs_batch, decoder_batch_length)

            outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size*seq_length, vocab_size]
            decoder_targets_batch = decoder_targets_batch.view(-1) # [batch_size*seq_length]
            loss = criterion(outputs, decoder_targets_batch)
            total_loss += loss.cpu().item()
    return total_loss / batch_num




def train():
    model = Transformer(en_total_words, MAX_SEQ_LENGTH, ch_total_words, MAX_SEQ_LENGTH, encoder_pad_idx=en_word_to_index['<PAD>'], decoder_pad_idx=ch_word_to_index['<PAD>'])
    model.to(device)
    # print(model)
    criterion = nn.CrossEntropyLoss(ignore_index=ch_word_to_index['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.
        batch_num = len(encoder_inputs_train) // BATCH_SIZE
        batch = get_batch(encoder_inputs_train, decoder_inputs_train, decoder_targets_trian, BATCH_SIZE)
        for i in range(batch_num):
            encoder_inputs_batch, decoder_inputs_batch, decoder_targets_batch, encoder_batch_length, decoder_batch_length = next(batch)
            encoder_inputs_batch = sequence_padding(encoder_inputs_batch, encoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)
            decoder_inputs_batch = sequence_padding(decoder_inputs_batch, decoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)
            decoder_targets_batch = sequence_padding(decoder_targets_batch, decoder_batch_length, max_seq_lenth=MAX_SEQ_LENGTH)

            outputs = model(encoder_inputs_batch, encoder_batch_length, decoder_inputs_batch, decoder_batch_length)

            outputs = outputs.view(-1, outputs.shape[-1])  # [batch_size*seq_length, vocab_size]
            decoder_targets_batch = decoder_targets_batch.view(-1) # [batch_size*seq_length]

            optimizer.zero_grad()
            loss = criterion(outputs, decoder_targets_batch)
            total_loss += loss.cpu().item()

            loss.backward()
            optimizer.step()

        eval = evaluate(model, encoder_inputs_test, decoder_inputs_test, decoder_targets_test, criterion, BATCH_SIZE)
        print('epoch %d, loss %.4f, eval %.4f, time %.2fmin' % (epoch + 1, total_loss / batch_num, eval, (time.time()-start)/60))
    torch.save(model.state_dict(), MODEL_PATH)


def test(english):
    encoder_inputs, encoder_inputs_length = generate_predict_sentence(english, en_word_to_index)
    # encoder_inputs = torch.from_numpy(encoder_inputs).to(device).long()
    # encoder_inputs_length = torch.from_numpy(encoder_inputs_length).to(device).long()
    encoder_inputs = sequence_padding(encoder_inputs, encoder_inputs_length, max_seq_lenth=MAX_SEQ_LENGTH)

    model = Transformer(en_total_words, MAX_SEQ_LENGTH, ch_total_words, MAX_SEQ_LENGTH,
                        encoder_pad_idx=en_word_to_index['<PAD>'], decoder_pad_idx=ch_word_to_index['<PAD>'])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    predict = model.translate(encoder_inputs, encoder_inputs_length)
    print(predict)
    predict = [ch_index_to_word[item] for item in predict]
    print(predict)


# train()
# english = 'I come from Chongqing!'
english = 'I love you'
test(english)