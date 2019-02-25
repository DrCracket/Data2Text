###############################################################################
# Provides functions to load and create datasets for the extraction module    #
###############################################################################

import torch
import tarfile
import pickle
import logging
from json import loads
from os import path, makedirs
from .data_structures import Vocab, OrderedCounter, ExtractorDataset
from .helper_funcs import get_candidate_rels, append_multilabeled_data


def preproc_extractor_data(corpus_type, folder, dataset_name, train_stats=None):
    location = f"{folder}/{dataset_name}.tar.bz2"

    with tarfile.open(location, 'r:bz2') as f:
        dataset = dict()
        for type_ in ["train", "valid", "test"]:
            dataset[type_] = loads(f.extractfile(f"{dataset_name}/{type_}.json").read())

    extracted_rel_tups = get_candidate_rels(dataset)

    # make vocab and get labels
    word_counter = OrderedCounter()
    for entry in extracted_rel_tups["train"]:
        for tup in entry:
            word_counter.update(tup[0])
        for k in list(word_counter.keys()):
            if word_counter[k] < 2:
                del word_counter[k]  # will replace w/ unk
    vocab = Vocab(word_counter.keys())
    labeldict = Vocab()
    # only use the labels that occur in the training dataset
    for entry in extracted_rel_tups["train"]:
        for tup in entry:
            for rel in tup[1]:
                labeldict.append(rel[2])

    sents, entdists, numdists, labels, len_entries, idx_list = [], [], [], [], [], []
    max_len = max((len(tup[0]) for entry in extracted_rel_tups[corpus_type] for tup in entry))
    for idx, entry in enumerate(extracted_rel_tups[corpus_type]):
        append_multilabeled_data(entry, sents, entdists, numdists, labels,
                                 vocab, labeldict, max_len, len_entries, idx_list, idx)

    # create tensors from lists
    sents_ts = torch.tensor(sents)
    entdists_ts = torch.tensor(entdists)
    numdists_ts = torch.tensor(numdists)

    # encode labels one hot style
    labels_ts = torch.zeros(len(sents), len(labeldict))
    for idx, labels in enumerate(labels):
        labels_ts[idx][labels] = 1

    # create the folder for the cached files if it does not exist
    if not path.exists(".cache/extractor"):
        makedirs(".cache/extractor")

    if corpus_type != "train":
        # load the train stats, that are used to remove unseen data from train and test set
        if path.exists(".cache/extractor/stats.pt"):
            # if they are not cached, create them
            train_stats = pickle.load(open(".cache/extractor/stats.pt", "rb"))
        else:
            _, _, _, train_stats, _ = preproc_extractor_data("train", folder, dataset_name)

        # clamp values so no other distances except the ones in the train set occur
        entdists_ts = entdists_ts.clamp(train_stats["min_entd"], train_stats["max_entd"])
        numdists_ts = numdists_ts.clamp(train_stats["min_numd"], train_stats["max_numd"])
    else:  # the current corpus_type is "train" so we need to create the train_stats
        min_entd, max_entd = entdists_ts.min().item(), entdists_ts.max().item()
        min_numd, max_numd = numdists_ts.min().item(), numdists_ts.max().item()

        # save shifted number to shift back when a content plan is generated
        entshift = min_entd
        numshift = min_numd

        # the number of entries the distance embeddings have to have
        ent_len = max_entd - min_entd + 1
        num_len = max_numd - min_numd + 1

        n_words = len(vocab)
        n_types = len(labeldict)

        train_stats = {"entshift": entshift, "numshift": numshift, "min_entd": min_entd,
                       "max_entd": max_entd, "min_numd": min_numd, "max_numd": max_numd,
                       "ent_len": ent_len, "num_len": num_len, "n_words": n_words,
                       "n_types": n_types}
        pickle.dump(train_stats, open(".cache/extractor/stats.pt", "wb"))

    # shift values to eliminate negative numbers
    entdists_ts -= train_stats["min_entd"]
    numdists_ts -= train_stats["min_numd"]

    # write dicts, lists and tensors to disk
    idx2word = dict(((v, k) for k, v in vocab.items()))
    idx2type = dict(((v, k) for k, v in labeldict.items()))
    torch.save(sents_ts, f".cache/extractor/{corpus_type}_sents.pt")
    torch.save(entdists_ts, f".cache/extractor/{corpus_type}_entdists.pt")
    torch.save(numdists_ts, f".cache/extractor/{corpus_type}_numdists.pt")
    torch.save(labels_ts, f".cache/extractor/{corpus_type}_labels.pt")
    pickle.dump(len_entries, open(f".cache/extractor/{corpus_type}_len_entries.pt", "wb"))
    pickle.dump(idx_list, open(f".cache/extractor/{corpus_type}_idx_list.pt", "wb"))
    pickle.dump(idx2word, open(".cache/extractor/vocab.pt", "wb"))
    pickle.dump(idx2type, open(".cache/extractor/labels.pt", "wb"))

    return (sents_ts, entdists_ts, numdists_ts, labels_ts), idx2word, idx2type, train_stats, len_entries, idx_list


def load_extractor_data(corpus_type, folder="boxscore-data", dataset="rotowire"):
    """Load a dataset e.g. for use with a dataloader"""
    try:
        sents = torch.load(f".cache/extractor/{corpus_type}_sents.pt")
        entdists = torch.load(f".cache/extractor/{corpus_type}_entdists.pt")
        numdists = torch.load(f".cache/extractor/{corpus_type}_numdists.pt")
        labels = torch.load(f".cache/extractor/{corpus_type}_labels.pt")
        len_entries = pickle.load(open(f".cache/extractor/{corpus_type}_len_entries.pt", "rb"))
        idx_list = pickle.load(open(f".cache/extractor/{corpus_type}_idx_list.pt", "rb"))
        idx2word = pickle.load(open(".cache/extractor/vocab.pt", "rb"))
        idx2type = pickle.load(open(".cache/extractor/labels.pt", "rb"))
        stats = pickle.load(open(".cache/extractor/stats.pt", "rb"))

    except FileNotFoundError:
        logging.warning(f"Failed to locate cached extractor {corpus_type} corpus!")
        logging.info(f"Genrating a new corpus...")
        type_data, idx2word, idx2type, stats, len_entries, idx_list = preproc_extractor_data(
            corpus_type, folder, dataset)
        sents, entdists, numdists, labels = type_data

    return ExtractorDataset(sents, entdists, numdists, labels, idx2word, idx2type, stats, len_entries, idx_list)
