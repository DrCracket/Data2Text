###############################################################################
# Information Extraction Module                                               #
###############################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Recall
from util.extractor import load_extractor_data
from abc import abstractmethod, ABC
from os import path, makedirs
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MarginalNLLLoss(nn.Module):
    """ The loss function proposed by Whiteman et al."""
    def forward(self, x, y):
        # calculate the log on all true labels
        logs = torch.where(y == 1, x.log(), torch.zeros(1))
        return -logs.mean()


class Extractor(nn.Module, ABC):
    def __init__(self, word_input_size, ent_dist_size, num_dist_size, word_hidden_size, dist_hidden_size):
        super().__init__()

        self.word_embedding = nn.Embedding(word_input_size, word_hidden_size)
        self.ent_embedding = nn.Embedding(ent_dist_size, dist_hidden_size)
        self.num_embedding = nn.Embedding(num_dist_size, dist_hidden_size)

    @abstractmethod
    def forward(self, sents, entdists, numdists):
        pass


class LSTMExtractor(Extractor):
    def __init__(self, word_input_size, ent_dist_size, num_dist_size, word_hidden_size=200, dist_hidden_size=100,
                 lstm_hidden_size=500, mlp_hidden_size=700, num_types=40, dropout=0.5):

        super().__init__(word_input_size, ent_dist_size, num_dist_size, word_hidden_size, dist_hidden_size)
        self.lstm_hidden_size = lstm_hidden_size
        input_width = word_hidden_size + 2 * dist_hidden_size

        self.rnn = nn.LSTM(input_width, lstm_hidden_size, batch_first=True, bidirectional=True)

        self.relu_mlp = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, None)),
            nn.Linear(2 * lstm_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(mlp_hidden_size, num_types),
            nn.Softmax(dim=2))

    def forward(self, sents, entdists, numdists):
        emb_sents = self.word_embedding(sents)
        emb_entdists = self.ent_embedding(entdists)
        emb_numdists = self.num_embedding(numdists)
        emb_cat = torch.cat((emb_sents, emb_entdists, emb_numdists), 2)

        # pack the padded sequence, use sents to calculate sequence lenth for
        # each batch element, because  entdist and numdist aren't zeropadded
        lengths = (sents > 0).sum(dim=1)
        # sort length tensor and batch  for pad_packed_sequence
        # (reminder: this hacky reordering won't be necessary in pytorch 1.1)
        lengths, perm_idx = lengths.sort(descending=True)
        emb_cat = emb_cat[perm_idx]

        lstm_input = pack_padded_sequence(emb_cat, lengths, batch_first=True)
        lstm_output, _ = self.rnn(lstm_input)
        padded_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # restore original ordering
        _, unperm_idx = perm_idx.sort()
        sorted_output = padded_output[unperm_idx]

        mlp_output = self.relu_mlp(sorted_output)
        soft_output = self.decoder(mlp_output)

        return soft_output.squeeze(1)


class CNNExtractor(Extractor):
    def __init__(self, word_input_size, ent_dist_size, num_dist_size, word_hidden_size=200, dist_hidden_size=100,
                 num_filters=200, mlp_hidden_size=500, num_types=40, dropout=0.5):

        super().__init__(word_input_size, ent_dist_size, num_dist_size, word_hidden_size, dist_hidden_size)
        input_width = word_hidden_size + 2 * dist_hidden_size

        self.mask_conv2 = nn.Conv2d(1, num_filters, (2, input_width), padding=(1, 0), bias=False)
        self.conv_kernel2 = nn.Sequential(
            nn.Conv2d(1, num_filters, (2, input_width), padding=(1, 0)),
            nn.ReLU())

        self.mask_conv3 = nn.Conv2d(1, num_filters, (3, input_width), padding=(2, 0), bias=False)
        self.conv_kernel3 = nn.Sequential(
            nn.Conv2d(1, num_filters, (3, input_width), padding=(2, 0)),
            nn.ReLU())

        self.mask_conv5 = nn.Conv2d(1, num_filters, (5, input_width), padding=(4, 0), bias=False)
        self.conv_kernel5 = nn.Sequential(
            nn.Conv2d(1, num_filters, (5, input_width), padding=(4, 0)),
            nn.ReLU())

        self.relu_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * num_filters, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(mlp_hidden_size, num_types),
            nn.Softmax(dim=2))

    def forward(self, sents, entdists, numdists):
        emb_sents = self.word_embedding(sents)
        emb_entdists = self.ent_embedding(entdists)
        emb_numdists = self.num_embedding(numdists)
        # add 1 dim for 2d convolution
        emb_cat = torch.cat((emb_sents, emb_entdists, emb_numdists), 2).unsqueeze(1)

        # mask padded values
        with torch.no_grad():
            pad_tensor = sents.view(sents.size(0), 1, -1, 1).expand_as(emb_cat).float()
            mask2 = self.mask_conv2(pad_tensor) == 0
            mask3 = self.mask_conv3(pad_tensor) == 0
            mask5 = self.mask_conv5(pad_tensor) == 0

        conv2 = F.adaptive_max_pool2d(self.conv_kernel2(emb_cat).masked_fill_(mask2, 0.0), (1, None))
        conv3 = F.adaptive_max_pool2d(self.conv_kernel3(emb_cat).masked_fill_(mask3, 0.0), (1, None))
        conv5 = F.adaptive_max_pool2d(self.conv_kernel5(emb_cat).masked_fill_(mask5, 0.0), (1, None))

        conv_cat = torch.cat((conv2, conv3, conv5), 1).view(conv2.size(0), 1, -1)

        output = self.decoder(self.relu_mlp(conv_cat))

        return output.squeeze(1)

###############################################################################
# Training & Evaluation functions                                             #
###############################################################################


def to_device(tensor_list):
    return [t.to(device, non_blocking=True) for t in tensor_list]


def train_extractor(batch_size=32, epochs=10, learning_rate=0.7, decay=0.5, clip=5, log_interval=1000, lstm=False):
    Model = LSTMExtractor if lstm else CNNExtractor
    data = load_extractor_data("train")
    loader = DataLoader(data, shuffle=True, batch_size=batch_size, pin_memory=torch.cuda.is_available())
    loss_fn = MarginalNLLLoss()

    extractor = Model(data.stats["n_words"], data.stats["ent_len"], data.stats["num_len"], num_types=data.stats["n_types"]).to(device)
    optimizer = optim.SGD(extractor.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)

    logging.info("Training a new extractor...")

    def _update(engine, batch):
        extractor.train()
        optimizer.zero_grad()
        b_sents, b_ents, b_nums, b_labs = to_device(batch)
        y_pred = extractor(b_sents, b_ents, b_nums)
        loss = loss_fn(y_pred, b_labs)
        loss.backward()
        nn.utils.clip_grad_value_(extractor.parameters(), clip)
        optimizer.step()
        return loss.item()

    trainer = Engine(_update)

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
    def _adapt_lr(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate(engine):
        eval_extractor(extractor)

    @trainer.on(Events.COMPLETED)
    def _test(engine):
        eval_extractor(extractor, test=True)

    trainer.run(loader, epochs)
    logging.info("Finished training process!")

    return extractor


def eval_extractor(extractor, test=False):
    loss_fn = MarginalNLLLoss()
    if test:
        used_set = "Test"
        loader = DataLoader(load_extractor_data("test"), batch_size=1000)
    else:
        used_set = "Validation"
        loader = DataLoader(load_extractor_data("valid"), batch_size=1000)

    def _update(engine, batch):
        """Transform the multi-label one-hot labels to multiclass indexes.
        I consider an example as correctly predicted when a label matches."""
        extractor.eval()
        with torch.no_grad():
            b_sents, b_ents, b_nums, b_labs = to_device(batch)
            y_pred = extractor(b_sents, b_ents, b_nums)
            loss = loss_fn(y_pred, b_labs)

            # transform the multi-label one-hot labels
            idxs_pred = y_pred.argmax(dim=1)
            idxs_lab = b_labs.argmax(dim=1)
            y = torch.zeros(len(b_labs), dtype=torch.long, device=device)
            for i in range(len(y)):
                y[i] = idxs_pred[i] if b_labs[i][idxs_pred[i]] == 1 else idxs_lab[i]

            return loss, y_pred, y

    evaluator = Engine(_update)
    accuracy = Accuracy(lambda output: output[1:])
    recall = Recall(lambda output: output[1:], average=True)

    accuracy.attach(evaluator, "accuracy")
    recall.attach(evaluator, "recall")

    @evaluator.on(Events.COMPLETED)
    def log_validation_results(engine):
        metrics = engine.state.metrics
        logging.info("{} Results - Avg accuracy: {:.2f}% Avg recall: {:.2f}%"
                     .format(used_set, 100 * metrics["accuracy"], 100 * metrics["recall"]))

    evaluator.run(loader)


def get_extractor(batch_size=32, epochs=10, learning_rate=0.7, decay=0.5, clip=5, log_interval=1000, lstm=False):
    prefix, Model = ("lstm", LSTMExtractor) if lstm else ("cnn", CNNExtractor)
    logging.info(f"Trying to load cached {prefix} extractor model...")

    if path.exists(f"models/{prefix}_extractor.pt"):
        data = load_extractor_data("train")
        extractor = Model(data.stats["n_words"], data.stats["ent_len"], data.stats["num_len"], num_types=data.stats["n_types"])
        extractor.load_state_dict(torch.load(f"models/{prefix}_extractor.pt", map_location="cpu"))
        logging.info("Success!")
    else:
        logging.warning("Failed to locate model.")
        if not path.exists("models"):
            makedirs("models")
        extractor = train_extractor(batch_size, epochs, learning_rate, decay, clip, log_interval, lstm)
        torch.save(extractor, f"models/{prefix}_extractor.pt")

    return extractor
