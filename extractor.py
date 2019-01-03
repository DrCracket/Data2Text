import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import ExtractorLoader


class Extractor(nn.Module):
    def __init__(self, word_input_size, dist_input_size, word_hidden_size=200, dist_hidden_size=100, num_filters=200):
        super(Extractor, self).__init__()
        conv_width = word_hidden_size + 2 * dist_hidden_size

        self.word_embedding = nn.Embedding(word_input_size, word_hidden_size)
        self.dist_embedding = nn.Embedding(dist_input_size, dist_hidden_size)

        # TODO: wide or narrow convolution?
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
            nn.Linear(3 * num_filters, 500),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(500, 40),
            nn.Softmax(dim=2))

    def forward(self, sent, entdist, numdist):
        emb_sent = self.word_embedding(sent)
        emb_entdist = self.dist_embedding(entdist)
        emb_numdist = self.dist_embedding(numdist)
        emb_cat = torch.cat((emb_sent, emb_entdist, emb_numdist), 2).unsqueeze(1)

        # mask padding values
        with torch.no_grad():
            pad_tensor = sent.view(1, 1, -1, 1).expand_as(emb_cat).float()
            mask2 = self.mask_conv2(pad_tensor) == 0
            mask3 = self.mask_conv3(pad_tensor) == 0
            mask5 = self.mask_conv5(pad_tensor) == 0

        conv2 = F.adaptive_max_pool2d(self.conv_kernel2(emb_cat).masked_fill_(mask2, 0.0), (1, None))
        conv3 = F.adaptive_max_pool2d(self.conv_kernel3(emb_cat).masked_fill_(mask3, 0.0), (1, None))
        conv5 = F.adaptive_max_pool2d(self.conv_kernel5(emb_cat).masked_fill_(mask5, 0.0), (1, None))

        conv_cat = torch.cat((conv2, conv3, conv5), 1).view(conv2.size(0), 1, -1)

        output = self.decoder(self.relu_mlp(conv_cat))

        return output


loader = ExtractorLoader()
extractor = Extractor(loader.n_words, loader.max_len * 2 - 1)

trsents, trlens, trentdists, trnumdists, trlabels = loader.trdata

# shift values to eliminate negative numbers
trentdists += loader.max_len - 1
trnumdists += loader.max_len - 1

print(extractor.forward(trsents[0:1], trentdists[0:1], trnumdists[0:1]))
