import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Recall
from data_utils import ExtractorDataset


class MarginalNLLLoss(nn.Module):
    def forward(self, x, y):
        # calculate the log on all true labels
        logs = torch.where(y == 1, x.log(), torch.tensor([0], dtype=torch.float))
        return -logs.mean()


class Extractor(nn.Module):
    def __init__(self, word_input_size, dist_input_size, word_hidden_size=200, dist_hidden_size=100, num_filters=200, num_types=40, dropout=0.5):
        super().__init__()
        conv_width = word_hidden_size + 2 * dist_hidden_size

        self.word_embedding = nn.Embedding(word_input_size, word_hidden_size)
        self.dist_embedding = nn.Embedding(dist_input_size, dist_hidden_size)

        self.mask_conv2 = nn.Conv2d(1, num_filters, (2, conv_width), padding=(1, 0), bias=False)
        self.conv_kernel2 = nn.Sequential(
            nn.Conv2d(1, num_filters, (2, conv_width), padding=(1, 0)),
            nn.ReLU())

        self.mask_conv3 = nn.Conv2d(1, num_filters, (3, conv_width), padding=(2, 0), bias=False)
        self.conv_kernel3 = nn.Sequential(
            nn.Conv2d(1, num_filters, (3, conv_width), padding=(2, 0)),
            nn.ReLU())

        self.mask_conv5 = nn.Conv2d(1, num_filters, (5, conv_width), padding=(4, 0), bias=False)
        self.conv_kernel5 = nn.Sequential(
            nn.Conv2d(1, num_filters, (5, conv_width), padding=(4, 0)),
            nn.ReLU())

        self.relu_mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * num_filters, 500),
            nn.ReLU(),
            nn.Dropout(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(500, num_types),
            nn.Softmax(dim=2))

    def forward(self, sents, entdists, numdists):
        emb_sents = self.word_embedding(sents)
        emb_entdists = self.dist_embedding(entdists)
        emb_numdists = self.dist_embedding(numdists)
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

    def extract_relations(self, dataset):
        relations = []

        for sent, entdist, numdist, _ in dataset:
            prediction = self(sent.unsqueeze(0), entdist.unsqueeze(0), numdist.unsqueeze(0))
            type_ = dataset.idx2type[prediction.argmax().item()]
            entity = []
            number = []
            for word, ent, num in zip(sent, entdist, numdist):
                if ent.item() + dataset.entshift == 0:
                    entity.append(dataset.idx2word[word.item()])
                if num.item() + dataset.numshift == 0:
                    number.append(dataset.idx2word[word.item()])
            relations.append([" ".join(entity), " ".join(number), type_])
        return relations


def train_extractor(batch_size=32, epochs=10, learning_rate=0.7, decay=0.5, clip=5, log_interval=1000):
    data = ExtractorDataset("train")
    loader = DataLoader(data, shuffle=True, batch_size=batch_size)

    extractor = Extractor(data.n_words, data.max_dist, num_types=data.n_types)
    optimizer = optim.SGD(extractor.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay)

    print("Training a new extractor...")

    def _update(engine, batch):
        extractor.train()
        optimizer.zero_grad()
        b_sents, b_ents, b_nums, b_labs = batch
        y_pred = extractor(b_sents, b_ents, b_nums)
        loss = MarginalNLLLoss()(y_pred, b_labs)
        loss.backward()
        torch.nn.utils.clip_grad_value_(extractor.parameters(), clip)
        optimizer.step()
        return loss.item()

    trainer = Engine(_update)

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

    @trainer.on(Events.EPOCH_COMPLETED)
    def adapt_lr(engine):
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        eval_extractor(extractor)

    @trainer.on(Events.COMPLETED)
    def test(engine):
        eval_extractor(extractor, test=True)

    trainer.run(loader, epochs)
    print("Finished training process!")

    return extractor


def eval_extractor(extractor, test=False):
    if test:
        used_set = "Test"
        loader = DataLoader(ExtractorDataset("test"), batch_size=1000)
    else:
        used_set = "Validation"
        loader = DataLoader(ExtractorDataset("valid"), batch_size=1000)

    def _update(engine, batch):
        """Transform the multi-label one-hot labels to multiclass indexes.
        I consider an example as correctly predicted when a label matches."""
        extractor.eval()
        with torch.no_grad():
            b_sents, b_ents, b_nums, b_labs = batch
            y_pred = extractor(b_sents, b_ents, b_nums)
            loss = MarginalNLLLoss()(y_pred, b_labs)

            # transform the multi-label one-hot labels
            idxs_pred = y_pred.argmax(dim=1)
            idxs_lab = b_labs.argmax(dim=1)
            y = torch.zeros(len(b_labs), dtype=torch.long)
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
        print("{} Results - Avg accuracy: {:.2f}% Avg recall: {:.2f}%"
              .format(used_set, 100 * metrics["accuracy"], 100 * metrics["recall"]))

    evaluator.run(loader)


def get_extractor(batch_size=32, epochs=10, learning_rate=0.7, decay=0.5, clip=5, log_interval=1000):
    try:
        print("Trying to load cached extractor model...")
        extractor = torch.load(".cache/extractor/extractor.pt")
        print("Success!")
    except FileNotFoundError:
        print("Failed to locate model.")
        extractor = train_extractor(batch_size, epochs, learning_rate, decay, log_interval)
        torch.save(extractor, ".cache/extractor/extractor.pt")

    return extractor


get_extractor()
