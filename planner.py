###############################################################################
# Content Selection and Planning Module                                       #
###############################################################################

from random import random
# from time import time
# from math import floor

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from data_utils import load_planner_data

# class ContentSelector(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(ContentSelector, self).__init__()
#         self.hidden_size = hidden_size
#
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.relu_mlp = nn.Sequential(
#             nn.Linear(4 * hidden_size, hidden_size),
#             nn.ReLU())
#         self.linear = nn.Linear(hidden_size, hidden_size)
#         self.sigmoid_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_size, hidden_size),
#             nn.Sigmoid())
#
#     def encode(self, records):
#         # size = (Batch x Records x 4 x hidden_size)
#         embedded = self.embedding(records)
#
#         # concatenate last dimension
#         # size = (Batch x Records x 4 * hidden_size)
#         emb_cat = embedded.view(embedded.size(0), embedded.size(1), -1)
#         # size = (Batch x Records x hidden_size)
#         emb_relu = self.relu_mlp(emb_cat)
#
#         # Content selection gate
#         # size = (Batch x hidden_size x Records)
#         emb_trans = torch.transpose(emb_relu, 1, 2)
#         # size = (Batch x Records x hidden_size)
#         emb_lin = self.linear(emb_relu)
#
#         # size = (Batch x Records x Records)
#         pre_attn = torch.bmm(emb_lin, emb_trans)
#         # diagonale should be zero. No attention for a record itself
#         mask = torch.diag(torch.ones(pre_attn.size(1), dtype=torch.uint8))
#         pre_attn = pre_attn.masked_fill_(mask, float("-Inf"))
#         attention = F.softmax(pre_attn, dim=2)
#
#         # apply attention
#         # size = (Batch x Records x hidden_size)
#         emb_att = torch.bmm(attention, emb_relu)
#         emb_gate = self.sigmoid_mlp(torch.cat((emb_relu, emb_att), 2))
#
#         output = emb_gate * emb_relu
#
#         return output


class ContentPlanner(nn.Module):
    def __init__(self, max_len, input_size, hidden_size=600):
        super(ContentPlanner, self).__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.mask = None
        self.selected_content = None

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.relu_mlp = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size),
            nn.ReLU())

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.sigmoid_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Sigmoid())

        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.softmax_mlp = nn.Sequential(
            nn.Linear(hidden_size, self.max_len),
            nn.LogSoftmax(dim=2))

    def forward(self, index, hidden, cell):
        """Content Planning. Uses attention to create pointers to the input records."""

        # size = (batch_size) => size = (batch_size x 1)
        index = index.unsqueeze(1)
        # update the mask with the input index, so that it won't get selected again
        self.mask = self.mask.scatter_(1, index, 1)
        # size = (batch_size x 1) => size = (batch_size x 1 x hidden_size)
        index = index.unsqueeze(2).repeat(1, 1, self.hidden_size)
        input_ = self.selected_content.gather(1, index)

        output, (hidden, cell) = self.rnn(input_, (hidden, cell))
        masked = output.masked_fill_(self.mask, float("-Inf"))
        attention = self.softmax_mlp(masked).unsqueeze(1)
        # update the mask with the selected input, so that it won't get selected again
        self.mask = self.mask.scatter_(1, attention.argmax(dim=1, keepdim=True), 1)

        return attention, hidden, cell

    def select_content(self, records):
        """Content selection gate. Determines importance vis-a-vis other records."""

        # size = (Batch x Records x 4 x hidden_size)
        embedded = self.embedding(records)
        # size = (Batch x Records x 4 * hidden_size)
        emb_cat = embedded.view(embedded.size(0), embedded.size(1), -1)
        # size = (Batch x Records x hidden_size)
        emb_relu = self.relu_mlp(emb_cat)
        # size = (Batch x hidden_size x Records)
        emb_trans = emb_relu.transpose(1, 2)
        # size = (Batch x Records x hidden_size)
        emb_lin = self.linear(emb_relu)

        # compute attention
        # size = (Batch x Records x Records)
        pre_attn = torch.bmm(emb_lin, emb_trans)
        # diagonale should be zero. No attention for a record itself
        # size = (Records x Records)
        diag_mask = torch.diag(torch.ones(pre_attn.size(1), dtype=torch.uint8))
        # padded values should also be masked out
        # size = (Batch x Records)
        pad_mask_pre, _ = (records == 0).max(dim=2)
        # size = (Batch x Records x Records)
        pad_mask_pre = pad_mask_pre.unsqueeze(1).repeat(1, records.size(1), 1)
        pad_mask = pad_mask_pre.transpose(1, 2) | pad_mask_pre
        mask = diag_mask | pad_mask
        pre_attn = pre_attn.masked_fill_(mask, float("-Inf"))
        attention = F.softmax(pre_attn, dim=2)

        # apply attention
        # size = (Batch x Records x hidden_size)
        emb_att = torch.bmm(attention, emb_relu)
        emb_gate = self.sigmoid_mlp(torch.cat((emb_relu, emb_att), 2))
        output = emb_gate * emb_relu

        return output

    def init_hidden(self, records):
        """Compute the initial hidden state and cell state of the Content Planning LSTM.
        Additionally initialize a mask to mask out the padded values of the LSTM inputs."""

        self.selected_content = self.select_content(records)
        self.mask = records.max(dim=1)[0] != 0
        # transpose first and second dim, because LSTM expects seq_len first
        hidden = torch.mean(self.selected_content, dim=1, keepdim=True).transpose(0, 1)
        cell = torch.zeros_like(hidden)

        return hidden, cell


def train_planner(extractor, epochs=25, learning_rate=0.15, decay=0.97, acc_val_init=0.1, teacher_forcing_ratio=0.5, log_interval=1000):
    data = load_planner_data("train", extractor)
    loader = DataLoader(data, shuffle=True, batch_size=1)  # online learning

    content_planner = ContentPlanner(data.records.size(1), len(data.idx2word))
    optimizer = optim.Adagrad(content_planner.parameters(), lr=learning_rate, lr_decay=decay, initial_accumulator_value=acc_val_init)

    print("Training a new extractor...")

    def update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        content_planner.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False

        records, content_plan = batch
        hidden, cell = content_planner.init_hidden(records)
        input_index = data.stats["BOS_index"]
        loss = 0

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for record_pointer in content_plan:
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)
                loss += F.nnl_loss(output, record_pointer)
                input_index = record_pointer  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for word_tensor in content_plan:
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)

                loss += F.nnl_loss(output, word_tensor)
                input_index = output.argmax(dim=1)
                if input_index.item() == data.stats["EOS_index"]:  # EOS token
                    break

        loss.backward()
        optimizer.step()
        return loss.item() / len(content_plan)  # normalize loss for logging

    trainer = Engine(update)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iteration = engine.state.iteration
        batch_size = loader.batch_size

        if iteration * batch_size % log_interval < batch_size:
            epoch = engine.state.epoch
            max_iters = len(loader)
            progress = 100 * iteration / (max_iters * epochs)
            loss = engine.state.output
            print("Training Progress {:.2f}% || Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}"
                  .format(progress, epoch, epochs, iteration % max_iters, max_iters, loss))

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def validate(engine):
    #     eval_extractor(extractor)

    # @trainer.on(Events.COMPLETED)
    # def test(engine):
    #     eval_extractor(extractor, test=True)

    trainer.run(loader, epochs)
    print("Finished training process!")

    return content_planner


# def train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio=0.5):
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     loss = 0
#     use_teacher_forcing = True if random() < teacher_forcing_ratio else False
#
#     for input_tensor, target_tensor in zip(input_tensors.split(1), target_tensors.split(1)):
#         encoder_output = encoder(input_tensor)
#
#         # Use SOS character as the initial input
#         decoder_input = encoder.embedding(torch.tensor([loader.word2index["SOS"]])).view(1, 1, -1)
#         decoder_hidden, decoder_cell = decoder.init(encoder_output)
#
#         if use_teacher_forcing:
#             # Teacher forcing: Feed the target as the next input
#             for word_tensor in target_tensor:
#                 decoder_output, decoder_hidden, decoder_cell = decoder(
#                     decoder_input, decoder_hidden, decoder_cell)
#                 loss += criterion(decoder_output, word_tensor)
#                 decoder_input = word_tensor  # Teacher forcing
#
#         else:
#             # Without teacher forcing: use its own predictions as the next input
#             for word_tensor in target_tensor:
#                 decoder_output, decoder_hidden, decoder_cell = decoder(
#                     decoder_input, decoder_hidden, decoder_cell)
#
#                 loss += criterion(decoder_output, word_tensor)
#
#                 index = torch.argmax(decoder_output, dim=2)
#
#                 if index < encoder_output.size(2):
#                     decoder_input = encoder_output[0, 0, index]
#                 else:  # predicted token is OOR because its EOS token
#                     break
#
#     loss.backward()
#
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item() / target_tensors.size(0) * target_tensors.size(1)
#
#
# def trainIters(encoder, decoder, input_tensors, target_tensors, print_every=1000, epochs=1, batch_size=1, learning_rate=0.01):
#     start = time()
#     print_loss_total = 0  # Reset every print_every
#     plot_loss_total = 0  # Reset every plot_every
#     n_iters = epochs * len(input_tensors) / batch_size
#
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
#     criterion = nn.NLLLoss()
#
#     for _ in range(epochs):
#         for iter, (input_tensor, target_tensor) in enumerate(zip(input_tensors.split(batch_size), target_tensors.split(batch_size))):
#
#             loss = train(input_tensor, target_tensor, encoder,
#                          decoder, encoder_optimizer, decoder_optimizer, criterion)
#             print_loss_total += loss
#             plot_loss_total += loss
#
#             if iter % print_every == 0:
#                 print_loss_avg = print_loss_total / print_every
#                 print_loss_total = 0
#                 print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                              iter, iter / n_iters * 100, print_loss_avg))
#
#
# def asMinutes(s):
#     m = floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
#
# def timeSince(since, percent):
#     now = time.time()
#     s = now - since
#     es = s / (percent)
#     rs = es - s
#     return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
#
#
# databases = ["test.json", "train.json", "valid.json"]
# loader = loader.load("rotowire/", databases[1])
# contentSelector = ContentSelector(loader.n_words, 5)
# contentPlanner = ContentPlanner(5, loader.samples.size(1))
#
# print(loader.samples.shape)
# trainIters(contentSelector, contentPlanner, loader.samples)
