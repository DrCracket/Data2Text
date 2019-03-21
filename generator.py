###############################################################################
# Text Generation Module                                                      #
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
from torch import optim
from random import random
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from util.generator import load_generator_data
from util.constants import PAD_WORD, BOS_WORD, EOS_WORD
from os import path, makedirs
from util.constants import device, TEXT_MAX_LENGTH
from util.helper_funcs import to_device
from util.metrics import CSMetric, RGMetric, COMetric, BleuScore


class TextGenerator(nn.Module):
    def __init__(self, record_encoder, word_input_size, word_hidden_size=600, hidden_size=600):
        super().__init__()
        self.encoded = None

        self.record_encoder = record_encoder
        self.embedding = nn.Embedding(word_input_size, word_hidden_size)
        self.encoder_rnn = nn.LSTM(self.record_encoder.hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_rnn = nn.LSTM(word_hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.tanh_mlp = nn.Sequential(
            nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False),
            nn.Tanh())
        self.soft_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, word_input_size),
            nn.LogSoftmax(dim=2))
        self.sig_copy = nn.Sequential(
            nn.Linear(2 * hidden_size, 1),
            nn.Sigmoid())

    def forward(self, word, hidden, cell):
        """
        Content Planning. Uses attention to create pointers to the input records.
        """
        # shape = (batch_size, 1, word_hidden_size)
        embedded = self.embedding(word).unsqueeze(1)
        # output.shape = (batch_size, seq_len, 2 * hidden_size)
        output, (new_hidden, new_cell) = self.decoder_rnn(embedded, (hidden, cell))
        # shape = (batch_size, 2 * hidden_size, seq_len)
        enc_lin = self.linear(self.encoded).transpose(1, 2)
        # shape = (batch_size, 1, seq_len)
        energy = torch.bmm(output, enc_lin)
        # shape = (batch_size, 1, 2 * hidden_size)
        selected = torch.bmm(F.softmax(energy, dim=2), self.encoded)

        att_hidden = self.tanh_mlp(torch.cat((output, selected), dim=2))
        out_prob = self.soft_mlp(att_hidden).squeeze(1)
        p_copy = self.sig_copy(output).squeeze(1)
        log_attention = F.log_softmax(energy, dim=2).squeeze(1)

        return out_prob, log_attention, p_copy, new_hidden, new_cell

    def init_hidden(self, records, content_plan):
        """
        Compute the initial hidden state and cell state of the Content Planning LSTM.
        Use an RNN to encode the record representations from the planning stage record encoder.
        """
        self.record_encoder(records)
        encoded_records = self.record_encoder.get_encodings(content_plan)
        # encoded.shape = (batch_size, seq_len, 2 * hidden_size)
        self.encoded, (hidden, cell) = self.encoder_rnn(encoded_records)
        return hidden, cell


###############################################################################
# Training & Evaluation functions                                             #
###############################################################################


def train_generator(extractor, content_planner, epochs=25, learning_rate=0.15, acc_val_init=0.1,
                    clip=7, teacher_forcing_ratio=1.0, log_interval=100):
    data = load_generator_data("train", extractor, content_planner, planner=True)
    loader = DataLoader(data, shuffle=True, pin_memory=torch.cuda.is_available())  # online learning

    generator = TextGenerator(copy.deepcopy(content_planner.record_encoder), len(data.idx2word)).to(device)
    optimizer = optim.Adagrad(generator.parameters(), lr=learning_rate, initial_accumulator_value=acc_val_init)

    logging.info("Training a new Text Generator...")

    def _update(engine, batch):
        """
        Update function for the Text Generation Module.
        Right now only online learning is supported
        """
        generator.train()
        optimizer.zero_grad()
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False
        text, copy_tgts, records, content_plan, copy_indices, copy_values = to_device(batch)
        # remove all the zero padded values from content plans
        content_plan = content_plan[:, :(content_plan > data.vocab[PAD_WORD]).sum(dim=1)]
        hidden, cell = generator.init_hidden(records, content_plan)
        text_iter, copy_index_iter = zip(text.t(), copy_tgts.t()), iter(copy_indices.t())
        input_word, _ = next(text_iter)

        loss = 0
        len_sequence = 0

        for word, copy_tgt in text_iter:
            if word.cpu() == data.vocab[PAD_WORD]:
                break
            out_prob, copy_prob, p_copy, hidden, cell = generator(
                input_word, hidden, cell)
            loss += F.binary_cross_entropy(p_copy, copy_tgt.view(-1, 1))
            if copy_tgt:
                loss += F.nll_loss(copy_prob, next(copy_index_iter))
            else:
                loss += F.nll_loss(out_prob, word)
            len_sequence += 1
            if use_teacher_forcing:
                input_word = word
            else:
                if p_copy.round():
                    input_word = copy_values[:, copy_prob.argmax(dim=1)].view(-1).detach()
                else:
                    input_word = out_prob.argmax(dim=1).detach()

        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), clip)
        optimizer.step()
        return loss.item() / len_sequence  # normalize loss for logging

    trainer = Engine(_update)
    # save the model every 4 epochs
    handler = ModelCheckpoint("data/model_cache", "generator", save_interval=4,
                              require_empty=False, save_as_state_dict=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"generator": generator})

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
        eval_generator(extractor, content_planner, generator)

    @trainer.on(Events.COMPLETED)
    def _test(engine):
        eval_generator(extractor, content_planner, generator, test=True)

    trainer.run(loader, epochs)
    logging.info("Finished training process!")

    if not path.exists("models"):
        makedirs("models")
    torch.save(generator.state_dict(), "models/text_generator.pt")

    return generator.cpu()


def eval_generator(extractor, content_planner, generator, test=False, planner=False):
    generator = generator.to(device)
    if test:
        used_set = "Test"
        data = load_generator_data("test", extractor, content_planner, planner=planner)
        loader = DataLoader(data)
    else:
        used_set = "Validation"
        data = load_generator_data("valid", extractor, content_planner, planner=planner)
        loader = DataLoader(data)
        prefix = "" if planner else " (without planner)"

    def _evaluate():
        generator.eval()
        cs_metric = CSMetric(extractor, "test" if test else "valid")
        rg_metric = RGMetric(extractor, "test" if test else "valid")
        co_metric = COMetric(extractor, "test" if test else "valid")
        bleu_metric = BleuScore()
        for idx, batch in enumerate(loader):
            gold_text, _, records, content_plan, _, copy_values = to_device(batch)
            # remove all the zero padded values from content plans
            content_plan = content_plan[:, :(content_plan > data.vocab[PAD_WORD]).sum(dim=1)]
            hidden, cell = generator.init_hidden(records, content_plan)
            input_word = torch.tensor([data.vocab[BOS_WORD]]).to(device, non_blocking=True)
            text = [input_word.item()]

            with torch.no_grad():
                while input_word.cpu() != data.vocab[EOS_WORD] and len(text) <= TEXT_MAX_LENGTH:
                    out_prob, copy_prob, p_copy, hidden, cell = generator(
                        input_word, hidden, cell)
                    if p_copy.round():
                        input_word = copy_values[:, copy_prob.argmax(dim=1)].view(1)
                    else:
                        input_word = out_prob.argmax(dim=1)
                    text.append(input_word.item())
            # convert indices to readable summaries
            gold_sum = [data.idx2word[idx.item()] for idx in gold_text[0] if
                        idx not in (data.vocab[BOS_WORD], data.vocab[EOS_WORD],
                        data.vocab[PAD_WORD])]
            gen_sum = [data.idx2word[idx] for idx in text[1:-1]]
            # feed summaries into all metrics
            cs_metric(gen_sum, gold_sum, data.idx_list[idx])
            co_metric(gen_sum, gold_sum, data.idx_list[idx])
            rg_metric(gen_sum, data.idx_list[idx])
            bleu_metric(gold_sum, gen_sum)

        logging.info("{}{} Results - CS Precision: {:.4f}%, CS Recall: {:.4f}%"
                     .format(used_set, prefix, *cs_metric.calculate()))
        logging.info("{}{} Results - RG Precision: {:.4f}%, RG #: {:.4f}"
                     .format(used_set, prefix, *rg_metric.calculate()))
        logging.info("{}{} Results - CO Damerau-Levenshtein Distance: {:.4f}%"
                     .format(used_set, prefix, co_metric.calculate()))
        logging.info("{}{} Results - BLEU Score: {:.4f}"
                     .format(used_set, prefix, bleu_metric.calculate()))

    _evaluate()


def generator_is_available():
    if path.exists("models/text_generator.pt"):
        logging.info("Found saved generator!")
        return True
    else:
        logging.warning("Failed to locate saved generator!")
        return False


def load_generator(extractor, content_planner):
    if path.exists("models/text_generator.pt"):
        data = load_generator_data("train", extractor, content_planner, planner=True)
        generator = TextGenerator(copy.deepcopy(content_planner.record_encoder), len(data.idx2word))
        generator.load_state_dict(torch.load("models/text_generator.pt", map_location="cpu"))
        return generator
