###############################################################################
# Various Data structures used throughout the network                         #
###############################################################################

from collections import Counter, OrderedDict
from torch.utils.data.dataset import Dataset
from .constants import PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class Vocab(dict):
    """"
    dict that uses the length of the dict as value for every mew key
    """

    def __init__(self, words=None, eos_and_bos=False):
        self[PAD_WORD] = len(self)  # 0
        self[UNK_WORD] = len(self)  # 1
        if eos_and_bos:
            self[BOS_WORD] = len(self)  # 2
            self[EOS_WORD] = len(self)  # 3
        if words:
            self.update(words)

    def __missing__(self, k):
        return self[UNK_WORD]

    def __setitem__(self, word, value):
        assert value == len(self), "Value and length have to be of same length!"
        if word not in self:
            super().__setitem__(word, value)

    def update(self, words):
        for word in words:
            self[word] = len(self)

    def append(self, word):
        self[word] = len(self)


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers the order elements are first encountered
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class ExtractorDataset(Dataset):
    """
    Dataset implementation for the content extractor
    """
    sents = None
    entdists = None
    numdists = None
    labels = None
    idx2word = None
    idx2type = None
    stats = None
    len_entries = None
    idx_list = None

    def __init__(self, sents, entdists, numdists, labels, idx2word, idx2type, stats, len_entries, idx_list):
        super().__init__()
        self.sents = sents
        self.entdists = entdists
        self.numdists = numdists
        self.labels = labels
        self.idx2word = idx2word
        self.idx2type = idx2type
        self.stats = stats
        self.len_entries = len_entries
        self.idx_list = idx_list

    def __getitem__(self, idx):
        return self.sents[idx], self.entdists[idx], self.numdists[idx], self.labels[idx]

    def __len__(self):
        return len(self.sents)

    def split(self, idx):
        return zip(self.sents.split(idx), self.entdists.split(idx), self.numdists.split(idx), self.labels.split(idx))


class SequenceDataset(Dataset):
    """
    Dataset implementation for the content planner
    """
    sequence = None
    content_plan = None
    idx2word = None
    vocab = None
    idx_list = None

    def __init__(self, sequence, content_plan, vocab, idx2word, idx_list):
        super().__init__()
        self.sequence = sequence
        self.content_plan = content_plan
        self.vocab = vocab
        self.idx2word = idx2word
        self.idx_list = idx_list

    def __getitem__(self, idx):
        return (self.sequence[idx], self.content_plan[idx])

    def __len__(self):
        return (len(self.sequence))


class CopyDataset(SequenceDataset):
    """
    Dataset implementation for the text generator
    """
    p_copy = None
    copy_indices = None
    copy_values = None
    records = None

    def __init__(self, sequence, p_copy, copy_indices, copy_values, records, content_plan, vocab, idx2word, idx_list):
        super().__init__(sequence, content_plan, vocab, idx2word, idx_list)
        self.p_copy = p_copy
        self.copy_indices = copy_indices
        self.copy_values = copy_values
        self.records = records

    def __getitem__(self, idx):
        return (self.sequence[idx], self.p_copy[idx], self.records[idx], self.content_plan[idx],
                self.copy_indices[idx], self.copy_values[idx])
