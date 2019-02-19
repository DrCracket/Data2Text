import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from random import random
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from data_utils import load_generator_data
from os import path, makedirs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextGenerator(nn.Module):
    def __init__(self, word_input_size, word_hidden_size=600, record_hidden_size=600, hidden_size=600):
        super().__init__()
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
        self.sig_copy = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid())

    def forward(self, word, hidden, cell):
        """Content Planning. Uses attention to create pointers to the input records."""
        # shape = (batch_size, 1, word_hidden_size)
        embedded = self.embedding(word).unsqueeze(1)
        # hidden.shape = (batch_size, 1, 2 * hidden_size)
        hidden, (_, cell) = self.decoder_rnn(embedded)
        # shape = (batch_size, 2 * hidden_size, seq_len)
        enc_lin = self.linear(self.encoded).transpose(1, 2)
        # shape = (batch_size, 1, seq_len)
        attention = F.softmax(torch.bmm(hidden, enc_lin), dim=2)
        # shape = (batch_size, 1, 2 * hidden_size)
        selected = torch.bmm(attention, self.encoded)

        new_hidden = self.tanh_mlp(torch.cat((hidden, selected), dim=2))
        out_prob = self.soft_mlp(new_hidden).squeeze(1)
        p_copy = self.sig_copy(new_hidden).squeeze(1)
        log_attention = attention.log().squeeze(1)

        # shape = (num_layers, num_directions, batch_size, hidden_size)
        new_hidden = new_hidden.view(1, 2, 1, -1)
        return out_prob, log_attention, p_copy, new_hidden, cell,

    def encode_recods(self, records):
        """Use an RNN to encode the record representations from the planning stage."""
        # encoded.shape = (batch_size, seq_len, 2 * hidden_size)
        encoded, (hidden, cell) = self.encoder_rnn(records)
        return encoded, hidden, cell

    def init_hidden(self, records):
        """Compute the initial hidden state and cell state of the Content Planning LSTM."""
        self.encoded, hidden, cell = self.encode_recods(records)
        return hidden, cell


def to_device(tensor_list):
     return [t.to(device, non_blocking=True) for t in tensor_list]


def train_generator(extractor, content_planner, epochs=25, learning_rate=0.15,
                    acc_val_init=0.1, clip=7, teacher_forcing_ratio=1.0, log_interval=100):
    data = load_generator_data("train", extractor, content_planner)
    loader = DataLoader(data, shuffle=True, batch_size=1, pin_memory=True)  # online learning

    generator = TextGenerator(len(data.idx2word)).to(device)
    optimizer = optim.Adagrad(generator.parameters(), lr=learning_rate, initial_accumulator_value=acc_val_init)

    print("Training a new Text Generator...")

    def _update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        generator.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False

        text, p_copy, content_plan, copy_indices, copy_values = to_device(batch)
        # remove all the zero padded values from the content plans
        non_zero = content_plan.nonzero()[:, 1].unique(sorted=True)
        non_zero = non_zero.view(1, -1, 1).repeat(1, 1, content_plan.size(2))
        hidden, cell = generator.init_hidden(content_plan.gather(1, non_zero))

        text_iterator, copy_word, copy_index = zip(text.t(), p_copy.t()), iter(copy_values.t()), iter(copy_indices.t())

        input_word, input_copy_prob = next(text_iterator)
        loss = 0
        len_sequence = 0
        
        if len(content_plan) > 0:
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for word, copy_tgt in text_iterator:
                    if word.cpu() == data.stats["PAD_INDEX"]:
                        break  # don't continue on the padded values
                    out_prob, copy_prob, p_copy, hidden, cell = generator(
                        input_word, hidden, cell)
                    loss += F.binary_cross_entropy(p_copy, copy_tgt.view(-1, 1))
                    if copy_tgt:
                        loss += F.nll_loss(copy_prob, next(copy_index))
                    else:
                        loss += F.nll_loss(out_prob, word)
                    len_sequence += 1
                    input_word = next(copy_word) if copy_tgt else word
            else:
                # Without teacher forcing: use its own predictions as the next input
                for word, copy_tgt in text_iterator:
                    if word.cpu() == data.stats["PAD_INDEX"]:
                        break
                    out_prob, copy_prob, p_copy, hidden, cell = generator(
                        input_word, hidden, cell)
                    loss += F.binary_cross_entropy(p_copy, copy_tgt.view(-1, 1))
                    if copy_tgt:
                        loss += F.nll_loss(copy_prob, next(copy_index))
                    else:
                        loss += F.nll_loss(out_prob, word)
                    len_sequence += 1
                    if p_copy > 0.5:
                        input_word = copy_values[:, copy_prob.argmax(dim=1)].view(1)
                    else:
                        input_word = out_prob.argmax(dim=1)

            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), clip)
            optimizer.step()
        return loss.item() / len_sequence  # normalize loss for logging

    trainer = Engine(_update)
    # save the model every 4 epochs
    handler = ModelCheckpoint('.cache/model_cache', 'generator', save_interval=4, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'generator': generator})

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

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def _validate(engine):
    #     eval_generator(extractor, generator)

    # @trainer.on(Events.COMPLETED)
    # def _test(engine):
    #     eval_generator(extractor, generator, test=True)

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

    def _update(engine, batch):
        """Update function for the Conent Selection & Planning Module.
        Right now only online learning is supported"""
        generator.eval()
        outputs = [data.stats["BOS_INDEX"]]
        with torch.no_grad():
            records, text = batch
            hidden, cell = generator.init_hidden(records)
            input_word = data.stats["BOS_INDEX"]

            while input_word != data.stats["EOS_INDEX"]:
                out_prob, copy_prob, p_copy, hidden, cell = generator(
                    input_word, hidden, cell)
                if p_copy:
                    outputs.append(data.copy2vocab[copy_prob])
                else:
                    outputs.append(data.copy2vocab[out_prob])

                input_word = data.copy2vocab[copy_prob.argmax(dim=1)] if p_copy else out_prob.argmax(dim=1)

        return torch.cat(outputs, dim=0), text

    evaluator = Engine(_update)
    # TODO: add Wiseman metrics when implemented
    # loss = Loss(text_metric)
    # loss.attach(evaluator, "text_metric")

    @evaluator.on(Events.COMPLETED)
    def _log_loss(engine):
        loss = engine.state.metrics["text-metric"]
        print("{} Results - Avg Loss: {:.4f}".format(used_set, loss))

    evaluator.run(loader)


def get_generator(extractor, content_planner, epochs=25, learning_rate=0.15,
                  acc_val_init=0.1, clip=7, teacher_forcing_ratio=1.0, log_interval=100):
    print("Trying to load cached content generator model...")
    if path.exists("models/content_generator.pt"):
        content_generator = torch.load("models/content_generator.pt")
        print("Success!")
    else:
        print("Failed to locate model.")
        if not path.exists("models"):
            makedirs("models")
        content_planner = train_generator(extractor, content_planner, epochs, learning_rate,
                                          acc_val_init, clip, teacher_forcing_ratio, log_interval)
        torch.save(content_generator, "models/content_generator.pt")

    return content_planner
