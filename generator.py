import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from random import random
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss
from data_utils import load_generator_data
from os import path, makedirs


class TextGenerator(nn.Module):
    def __init__(self, word_input_size, word_hidden_size=600, record_hidden_size=600, hidden_size=600):
        self.encoded = None

        self.embedding = nn.Embedding(word_input_size, word_hidden_size)
        self.encoder_rnn = nn.LSTM(record_hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(word_hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.tanh_mlp = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size),
            nn.Tanh())
        self.soft_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, word_input_size),
            nn.LogSoftmax(dim=2))

    def forward(self, word, hidden, cell):
        """Content Planning. Uses attention to create pointers to the input records."""
        # shape = (batch_size x 1 x word_hidden_size)
        embedded = self.embedding(word)
        # output.shape = (batch_size x 1 x 2 * hidden_size)
        output, (hidden, cell) = self.decoder_rnn(embedded)
        # shape = (batch_size, 2 * hidden_size, seq_len)
        enc_lin = self.linear(self.encoded).transpose(1, 2)
        # shape = (batch_size, 1, seq_len)
        attention = F.softmax(torch.bmm(output, enc_lin))
        # shape = (batch_size, 1, 2 * hidden_size)
        selected = torch.bmm(attention, self.encoded)

        new_hidden = self.tanh_mlp(torch.cat((output, selected), dim=2))
        prediction = self.soft_mlp(new_hidden)

        # shape = (num_layers, num_directions, batch_size, hidden_size)
        new_hidden = new_hidden.view(1, 2, 1, -1)

        return prediction, new_hidden, cell

    def encode_recods(self, records):
        """Content selection gate. Determines importance vis-a-vis other records."""
        # encoded.shape = (batch_size, seq_len, 2 * hidden_size)
        encoded, (hidden, cell) = self.encoder_rnn(records)
        return encoded, hidden, cell

    def init_hidden(self, records):
        """Compute the initial hidden state and cell state of the Content Planning LSTM.
        Additionally initialize a mask to mask out the padded values of the LSTM inputs."""
        self.encoded, hidden, cell = self.encode_recods(records)
        return hidden, cell


def train_generator(extractor, content_planner, epochs=25, learning_rate=0.15, acc_val_init=0.1, clip=7, teacher_forcing_ratio=0.5, log_interval=100):
    data = load_generator_data("train", extractor, content_planner)
    loader = DataLoader(data, shuffle=True, batch_size=1)  # online learning

    generator = TextGenerator(len(data.idx2word))
    optimizer = optim.Adagrad(generator.parameters(), lr=learning_rate, initial_accumulator_value=acc_val_init)

    print("Training a new Content Planner...")

    def _update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        generator.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False

        records, text = batch
        hidden, cell = generator.init_hidden(records)
        text_iterator = iter(text.t())
        input_word = next(text_iterator)
        loss = 0
        len_sequence = 0

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for word in text_iterator:
                if word == data.stats["PAD_INDEX"]:
                    break  # don't continue on the padded values
                output, hidden, cell = generator(
                    input_word, hidden, cell)
                loss += F.nll_loss(output, word)
                len_sequence += 1
                input_word = word
        else:
            # Without teacher forcing: use its own predictions as the next input
            for word in text_iterator:
                if word == data.stats["PAD_INDEX"]:
                    break
                output, hidden, cell = generator(
                    input_word, hidden, cell)
                loss += F.nll_loss(output, word)
                len_sequence += 1
                input_word = output.argmax(dim=1)

        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), clip)
        optimizer.step()
        return loss.item() / len_sequence  # normalize loss for logging

    trainer = Engine(_update)
    # save the model every 4 epochs
    handler = ModelCheckpoint('.cache/model_cache', 'planner', save_interval=4, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'planner': generator})

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
        eval_generator(extractor, generator)

    @trainer.on(Events.COMPLETED)
    def _test(engine):
        eval_generator(extractor, generator, test=True)

    trainer.run(loader, epochs)
    print("Finished training process!")

    return generator


def eval_generator(extractor, content_planner, generator, test=False):
    if test:
        used_set = "Test"
        data = load_generator_data("test", extractor, content_planner)
        loader = DataLoader(data, batch_size=1)
    else:
        used_set = "Validation"
        data = load_generator_data("valid", extractor, content_planner)
        loader = DataLoader(data, batch_size=1)

    def update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        generator.eval()
        outputs = list()
        labels = list()
        with torch.no_grad():
            records, content_plan = batch
            hidden, cell = generator.init_hidden(records)
            content_plan_iterator = iter(content_plan.t())
            input_index = next(content_plan_iterator)

            for record_pointer in content_plan_iterator:
                if record_pointer == data.stats["PAD_INDEX"]:
                    break
                output, hidden, cell = content_planner(
                    input_index, hidden, cell)
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
        print("{} Results - Avg Loss: {:.4f}".format(used_set, loss))

    evaluator.run(loader)


def get_extractor(extractor, content_planner, epochs=25, learning_rate=0.15, acc_val_init=0.1, clip=7, teacher_forcing_ratio=0.5, log_interval=100):
    print("Trying to load cached content selection & planning model...")
    if path.exists("models/content_planner.pt"):
        content_planner = torch.load("models/content_planner.pt")
        print("Success!")
    else:
        print("Failed to locate model.")
        if not path.exists("models"):
            makedirs("models")
        content_planner = train_generator(extractor, content_planner, epochs, learning_rate, acc_val_init, clip, teacher_forcing_ratio, log_interval)
        torch.save(content_planner, "models/content_planner.pt")

    return content_planner
