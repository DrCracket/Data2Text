###############################################################################
# Content Selection and Planning Module                                       #
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from random import random
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from data_utils import load_planner_data


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
        self.logits_mlp = nn.Linear(hidden_size, self.max_len)

    def forward(self, index, hidden, cell):
        """Content Planning. Uses attention to create pointers to the input records."""

        # size = (batch_size) => size = (batch_size x 1)
        index = index.unsqueeze(1)
        # update the mask with the input index, so that it won't get selected again
        self.mask = self.mask.scatter_(1, index, 1)
        # size = (batch_size x 1) => size = (batch_size x 1 x hidden_size)
        index = index.unsqueeze(2).repeat(1, 1, self.hidden_size)
        input_ = self.selected_content.gather(1, index)

        # size = (batch_size x 1 x hidden_size)
        output, (hidden, cell) = self.rnn(input_, (hidden, cell))
        # size = (batch_size x 1 x max_len)
        logits = self.logits_mlp(output)
        # size = (batch_size x 1 x max_len), clone for gradient computation
        masked = logits.masked_fill(self.mask.clone(), float("-Inf"))
        # size = (batch_size x max_len)
        attention = F.log_softmax(masked, dim=2).squeeze(1)

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
        pad_mask_pre = records.max(dim=2)[0] == 0
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
        self.mask = records.max(dim=2)[0] == 0
        # transpose first and second dim, because LSTM expects seq_len first
        hidden = torch.mean(self.selected_content, dim=1, keepdim=True).transpose(0, 1)
        cell = torch.zeros_like(hidden)

        return hidden, cell


def train_planner(extractor, epochs=25, learning_rate=0.15, decay=0.97, acc_val_init=0.1, teacher_forcing_ratio=0.5, log_interval=100):
    data = load_planner_data("train", extractor)
    loader = DataLoader(data, shuffle=True, batch_size=1)  # online learning

    content_planner = ContentPlanner(data.records.size(1), len(data.idx2word))
    optimizer = optim.Adagrad(content_planner.parameters(), lr=learning_rate, lr_decay=decay, initial_accumulator_value=acc_val_init)

    print("Training a new Content Planner...")

    def _update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        content_planner.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False

        records, content_plan = batch
        hidden, cell = content_planner.init_hidden(records)
        content_plan_iterator = iter(content_plan.t())
        input_index = next(content_plan_iterator)
        loss = 0
        len_sequence = 0

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for record_pointer in content_plan_iterator:
                if record_pointer == data.stats["PAD_INDEX"]:
                    break  # don't continue on the padded values
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)
                loss += F.nll_loss(output, record_pointer)
                len_sequence += 1
                input_index = record_pointer

        else:
            # Without teacher forcing: use its own predictions as the next input
            for record_pointer in content_plan_iterator:
                if record_pointer == data.stats["PAD_INDEX"]:
                    break
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)
                loss += F.nll_loss(output, record_pointer)
                len_sequence += 1
                input_index = output.argmax(dim=1)
                if input_index.item() == data.stats["EOS_INDEX"]:
                    break

        loss.backward()
        optimizer.step()
        return loss.item() / len_sequence  # normalize loss for log

    trainer = Engine(_update)
    # save the model every 4 epochs
    handler = ModelCheckpoint('.cache/model_cache', 'planner', save_interval=4, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'planner': content_planner})

    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_training_loss(engine):
        iteration = engine.state.iteration
        batch_size = loader.batch_size

        if iteration * batch_size % log_interval < batch_size:
            epoch = engine.state.epoch
            max_iters = len(loader)
            progress = 100 * iteration / (max_iters * epochs)
            loss = engine.state.output
            print("Training Progress {:.2f}% || Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}"
                  .format(progress, epoch, epochs, iteration % max_iters, max_iters, loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate(engine):
        eval_planner(extractor, content_planner)

    @trainer.on(Events.COMPLETED)
    def _test(engine):
        eval_planner(extractor, content_planner, test=True)

    trainer.run(loader, epochs)
    print("Finished training process!")

    return content_planner


def eval_planner(extractor, content_planner, teacher_forcing_ratio=0.5, test=False):
    if test:
        used_set = "Test"
        data = load_planner_data("test", extractor)
        loader = DataLoader(data, batch_size=1)
    else:
        used_set = "Validation"
        data = load_planner_data("test", extractor)
        loader = DataLoader(data, batch_size=1)

    correct = 0
    total = 0

    def update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        content_planner.eval()
        nonlocal correct
        nonlocal total
        with torch.no_grad():
            use_teacher_forcing = True if random() < teacher_forcing_ratio else False

            records, content_plan = batch
            hidden, cell = content_planner.init_hidden(records)
            content_plan_iterator = iter(content_plan.t())
            input_index = next(content_plan_iterator)

            for record_pointer in content_plan_iterator:
                if record_pointer == data.stats["PAD_INDEX"]:
                    break
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)
                if record_pointer == output.argmax(dim=1):
                    correct += 1
                total += 1
                if use_teacher_forcing:
                    input_index = record_pointer
                else:
                    input_index = output.argmax(dim=1)

    evaluator = Engine(update)

    @evaluator.on(Events.COMPLETED)
    def log_accuracy(engine):
        print("{} Results - Avg accuracy: {:.2f}%"
              .format(used_set, correct / total))

    evaluator.run(loader)
