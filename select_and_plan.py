###############################################################################
#                  Content selection and Planning module                      #
###############################################################################

from random import random
from time import time
from math import floor
# from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import loader


class ContentSelector(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContentSelector, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.linear1 = nn.Linear(4*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)

        # concatenate last dimension
        embedded = embedded.view(embedded.size(0), embedded.size(1), -1)
        r_j = F.relu(self.linear1(embedded))

        # Content selection gate
        r_k = torch.transpose(r_j, 1, 2)
        rw_j = self.linear2(r_j)

        pre_attn = torch.bmm(rw_j, r_k)
        # diagonale should be zero, bc no attention for a record itself
        mask = torch.diag(torch.ones(pre_attn.size(1), dtype=torch.uint8))
        pre_attn = pre_attn.masked_fill_(mask, float("-Inf"))
        attention = F.softmax(pre_attn, dim=2)

        # apply attention
        c_j = torch.bmm(attention, r_j)
        r_att = self.linear3(torch.cat((r_j, c_j), 2))

        g_j = torch.sigmoid(r_att)
        r_jcs = g_j * r_j

        return r_jcs


class ContentPlanner(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(ContentPlanner, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # add one to the output size for the eos token
        self.linear = nn.Linear(hidden_size, self.max_len + 1)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        attention = F.softmax(self.linear(output), dim=2)
        return attention, hidden, cell

    def init(self, encoder_outputs):
        self.encoder_outputs = encoder_outputs
        hidden = torch.mean(self.encoder_outputs, dim=1,
                            keepdim=True).permute(1, 0, 2)
        cell = torch.zeros_like(hidden)
        return hidden, cell


def train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    use_teacher_forcing = True if random() < teacher_forcing_ratio else False

    for input_tensor, target_tensor in zip(input_tensors.split(1), target_tensors.split(1)):
        encoder_output = encoder(input_tensor)

        # Use SOS character as the initial input
        decoder_input = encoder.embedding(torch.tensor([loader.word2index["SOS"]])).view(1, 1, -1)
        decoder_hidden, decoder_cell = decoder.init(encoder_output)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for word_tensor in target_tensor:
                decoder_output, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell)
                loss += criterion(decoder_output, word_tensor)
                decoder_input = word_tensor  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for word_tensor in target_tensor:
                decoder_output, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell)

                loss += criterion(decoder_output, word_tensor)

                index = torch.argmax(decoder_output, dim=2)

                if index < encoder_output.size(2):
                    decoder_input = encoder_output[0, 0, index]
                else:  # predicted token is OOR because its EOS token
                    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensors.size(0)*target_tensors.size(1)


def trainIters(encoder, decoder, input_tensors, target_tensors, print_every=1000, epochs=1, batch_size=1, learning_rate=0.01):
    start = time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    n_iters = epochs * len(input_tensors) / batch_size

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for _ in range(epochs):
        for iter, (input_tensor, target_tensor) in enumerate(zip(input_tensors.split(batch_size), target_tensors.split(batch_size))):

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))


def asMinutes(s):
    m = floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


databases = ["test.json", "train.json", "valid.json"]
loader = loader.load("rotowire/", databases[1])
contentSelector = ContentSelector(loader.n_words, 5)
contentPlanner = ContentPlanner(5, loader.samples.size(1))

print(loader.samples.shape)
#trainIters(contentSelector, contentPlanner, loader.samples)
