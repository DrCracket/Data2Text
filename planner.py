###############################################################################
# Content Selection and Planning Module                                       #
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch import optim
from random import random
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from util.planner import load_planner_data
from util.constants import PAD_WORD
from os import path, makedirs
from util.constants import device
from util.helper_funcs import to_device


class ContentPlanner(nn.Module):
    def __init__(self, input_size, hidden_size=600):
        super(ContentPlanner, self).__init__()
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
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, index, hidden, cell):
        """
        Content Planning. Uses attention to create pointers to the input records.
        """
        # size = (batch_size) => size = (batch_size, 1, hidden_size)
        index = index.view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        input_ = self.selected_content.gather(1, index)

        # size = (batch_size, 1, hidden_size)
        output, (hidden, cell) = self.rnn(input_, (hidden, cell))
        # size = (batch_size, hidden_size, records)
        content_tp = self.linear2(self.selected_content).transpose(1, 2)
        # size = (batch_size, 1, records)
        logits = torch.bmm(output, content_tp)
        # size = (batch_size, records)
        attention = F.log_softmax(logits, dim=2).squeeze(1)

        return attention, hidden, cell

    def select_content(self, records):
        """
        Content selection gate. Determines importance vis-a-vis other records.
        """
        # size = (Batch, Records, 4, hidden_size)
        embedded = self.embedding(records)
        # size = (Batch, Records, 4 * hidden_size)
        emb_cat = embedded.view(embedded.size(0), embedded.size(1), -1)
        # size = (Batch, Records, hidden_size)
        emb_relu = self.relu_mlp(emb_cat)

        # compute attention
        # size = (Batch, hidden_size, Records)
        emb_lin = self.linear(emb_relu).transpose(1, 2)
        # size = (Batch, Records, Records)
        logits = torch.bmm(emb_relu, emb_lin)
        attention = F.softmax(logits, dim=2)

        # apply attention
        # size = (Batch, Records, hidden_size)
        emb_att = torch.bmm(attention, emb_relu)
        emb_gate = self.sigmoid_mlp(torch.cat((emb_relu, emb_att), 2))
        output = emb_gate * emb_relu

        return output

    def init_hidden(self, records):
        """
        Compute the initial hidden state and cell state of the Content Planning LSTM.
        Additionally initialize a mask to mask out the padded values of the LSTM inputs.
        """
        self.selected_content = self.select_content(records)
        self.mask = records.max(dim=2)[0] == 0
        # transpose first and second dim, because LSTM expects seq_len first
        hidden = torch.mean(self.selected_content, dim=1, keepdim=True).transpose(0, 1)
        cell = torch.zeros_like(hidden)

        return hidden, cell

###############################################################################
# Training & Evaluation functions                                             #
###############################################################################


def train_planner(extractor, epochs=25, learning_rate=0.01, acc_val_init=0.1,
                  clip=7, teacher_forcing_ratio=1.0, log_interval=100):
    data = load_planner_data("train", extractor)
    loader = DataLoader(data, shuffle=True, pin_memory=torch.cuda.is_available())  # online learning

    content_planner = ContentPlanner(len(data.idx2word)).to(device)
    optimizer = optim.Adagrad(content_planner.parameters(), lr=learning_rate, initial_accumulator_value=acc_val_init)

    logging.info("Training a new Content Planner...")

    def _update(engine, batch):
        """
        Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported
        """

        content_planner.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False

        records, content_plan = to_device(batch)
        hidden, cell = content_planner.init_hidden(records)
        content_plan_iterator = iter(content_plan.t())
        input_index = next(content_plan_iterator)
        loss = 0
        len_sequence = 0

        for record_pointer in content_plan_iterator:
            if record_pointer.cpu() == data.vocab[PAD_WORD]:
                break
            output, hidden, cell = content_planner(input_index, hidden, cell)
            loss += F.nll_loss(output, record_pointer)
            len_sequence += 1
            if use_teacher_forcing:
                input_index = record_pointer
            else:
                input_index = output.argmax(dim=1).detach()

        loss.backward()
        nn.utils.clip_grad_norm_(content_planner.parameters(), clip)
        optimizer.step()
        return loss.item() / len_sequence  # normalize loss for logging

    trainer = Engine(_update)
    # save the model every 4 epochs
    handler = ModelCheckpoint(".cache/model_cache", "planner", save_interval=4,
                              require_empty=False, save_as_state_dict=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"planner": content_planner})

    @trainer.on(Events.ITERATION_COMPLETED)
    def _log_training_loss(engine):
        iteration = engine.state.iteration
        batch_size = loader.batch_size

        if iteration * batch_size % log_interval < batch_size:
            epoch = engine.state.epoch
            max_iters = len(loader)
            progress = 100 * iteration / (max_iters * epochs)
            loss = engine.state.output
            logging.info("Training Progress {:.2f}% || Epoch: {}/{}, Iteration: {}/{}, Loss: {:.4f}"
                         .format(progress, epoch, epochs, iteration % max_iters, max_iters, loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate(engine):
        eval_planner(extractor, content_planner)

    @trainer.on(Events.COMPLETED)
    def _test(engine):
        eval_planner(extractor, content_planner, test=True)

    trainer.run(loader, epochs)
    logging.info("Finished training process!")

    return content_planner.cpu()


def eval_planner(extractor, content_planner, test=False):
    content_planner = content_planner.to(device)
    if test:
        used_set = "Test"
        data = load_planner_data("test", extractor)
        loader = DataLoader(data, batch_size=1)
    else:
        used_set = "Validation"
        data = load_planner_data("valid", extractor)
        loader = DataLoader(data, batch_size=1)

    def update(engine, batch):
        """
        Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported
        """

        content_planner.eval()
        outputs = list()
        labels = list()

        with torch.no_grad():
            records, content_plan = to_device(batch)
            hidden, cell = content_planner.init_hidden(records)
            content_plan_iterator = iter(content_plan.t())
            input_index = next(content_plan_iterator)

            for record_pointer in content_plan_iterator:
                if record_pointer.cpu() == data.vocab[PAD_WORD]:
                    break
                output, hidden, cell = content_planner(input_index, hidden, cell)
                outputs.append(output)
                labels.append(record_pointer)
                input_index = output.argmax(dim=1)

        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

    evaluator = Engine(update)
    loss = Loss(F.nll_loss)
    loss.attach(evaluator, "loss")

    @evaluator.on(Events.COMPLETED)
    def log_loss(engine):
        loss = engine.state.metrics["loss"]
        logging.info("{} Results - Avg Loss: {:.4f}".format(used_set, loss))

    evaluator.run(loader)


def get_planner(extractor, epochs=25, learning_rate=0.01, acc_val_init=0.1,
                clip=7, teacher_forcing_ratio=1.0, log_interval=100):
    logging.info("Trying to load cached content selection & planning model...")
    if path.exists("models/content_planner.pt"):
        data = load_planner_data("train", extractor)
        content_planner = ContentPlanner(len(data.idx2word))
        content_planner.load_state_dict(torch.load("models/content_planner.pt", map_location="cpu"))
        logging.info("Success!")
    else:
        logging.warning("Failed to locate model.")
        if not path.exists("models"):
            makedirs("models")
        content_planner = train_planner(extractor, epochs, learning_rate, acc_val_init,
                                        clip, teacher_forcing_ratio, log_interval)
        torch.save(content_planner.state_dict(), "models/content_planner.pt")

    return content_planner
