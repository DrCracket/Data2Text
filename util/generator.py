###############################################################################
# Provides functions to load and create datasets for the                      #
# text generation module                                                      #
###############################################################################

import tarfile
import pickle
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
from nltk import sent_tokenize
from word2number import w2n
from os import path, makedirs
from json import loads
from .constants import number_words
from .helper_funcs import annoying_number_word, extract_entities, extract_numbers
from .planner import load_planner_data
from .constants import PAD_WORD, BOS_WORD, EOS_WORD, MAX_CONTENT_PLAN_LENGTH
from .data_structures import OrderedCounter, Vocab, CopyDataset


def make_content_plan(planner, dataset):
    """Generate a content plan with a trained content planner for the generator."""
    dim1, dim2, dim3 = dataset.sequence.size(0), dataset.sequence.size(1), planner.hidden_size
    # size = (#entries, records, hidden_size)
    content_plans = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dim3)
    record_indices = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dtype=torch.long)
    planner.eval()

    with torch.no_grad():
        for dim1 in range(len(dataset)):
            records, _ = dataset[dim1]
            hidden, cell = planner.init_hidden(records.unsqueeze(0))
            record_index = dataset.vocab[BOS_WORD]
            dim2 = 0
            while not record_index == dataset.vocab[EOS_WORD]:
                output, hidden, cell = planner(record_index, hidden, cell)
                # in 0.002% of all cases the content plan would be empty. To prevent that use the next likeliest
                # record in the distribution that isn't a special record like BOS, EOS, PAD
                if dim2 == 0:
                    _, top = torch.topk(output, 5, dim=1)
                    for index in top[0]:
                        if index > 4:
                            record_index = index
                            break
                else:
                    record_index = output.argmax(dim=1)
                if record_index > 4:  # not PAD, PAD, BOS, EOS
                    # size = (1) => size = (1, 1, hidden_size)
                    idx = record_index.view(-1, 1, 1).repeat(1, 1, planner.hidden_size)
                    content_plans[dim1][dim2] = planner.selected_content.gather(1, idx)
                    record_indices[dim1][dim2] = record_index
                    # stop when content_planner is to long
                    if dim2 < MAX_CONTENT_PLAN_LENGTH - 1:
                        dim2 += 1
                    else:
                        break

    return content_plans, record_indices


def get_copy_probs(summary, entry_indices, records, vocab, idx2word):
    """ Extract the probabilities of which words are copied from the dataset"""
    all_ents = set()
    all_sents = list()
    all_p_copy = list()
    copy_indices = list()
    copy_values = list()

    for index in entry_indices:
        if index == vocab[PAD_WORD]:
            break
        all_ents.add(idx2word[records[index][0].item()])

    entset = set()
    for entity in all_ents:
        for piece in entity.split(" "):
            if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                entset.add(piece)
    all_ents = all_ents | entset

    for sent in sent_tokenize(" ".join(summary)):
        tokes = list()
        split_sent = sent.split(" ")
        for i, word in enumerate(split_sent):
            if word in number_words and not annoying_number_word(split_sent, i):
                j = 1
                while i + j < len(split_sent) and split_sent in number_words and not annoying_number_word(split_sent,
                                                                                                          i + j):
                    j += 1
                tokes.append(str(w2n.word_to_num(" ".join(split_sent[i:i + j]))))
                i += j
            else:
                tokes.append(word)
        p_copy = [0] * len(tokes)
        ents = extract_entities(tokes, all_ents, list())
        nums = extract_numbers(tokes)
        for ent in ents:
            for num in nums:
                for i, index in enumerate(entry_indices):
                    if index == vocab[PAD_WORD]:
                        break
                    entity = idx2word[records[index][0].item()]
                    identifiers = {entity}
                    value = idx2word[records[index][2].item()]
                    for piece in entity.split():
                        if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                            identifiers.add(piece)
                    if ent[2] in identifiers and num[2] == int(value):
                        p_copy[num[0]:num[1]] = [1]
                        copy_indices.append(i)
                        copy_values.append(idx2word[records[index][2].item()])
                        break
        all_sents.extend(tokes)
        all_p_copy.extend(p_copy)

    return all_sents, all_p_copy, copy_indices, copy_values


def preproc_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    plan_dataset = load_planner_data(corpus_type, extractor, folder, dataset)
    content_plans, record_indices = make_content_plan(planner, plan_dataset)
    summaries = list()
    all_p_copy = list()
    all_copy_indices = list()
    all_copy_values = list()

    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())

    for idx in range(len(plan_dataset)):
        records, _ = plan_dataset[idx]
        entry_indices = record_indices[idx]
        summary = raw_dataset[idx]["summary"]

        summary, p_copy, copy_indices, copy_values = get_copy_probs(summary, entry_indices,
                                                                    records, plan_dataset.vocab, plan_dataset.idx2word)
        summary.insert(0, BOS_WORD)
        summary.append(EOS_WORD)
        summaries.append(summary)
        p_copy.insert(0, 0)
        p_copy.append(0)
        all_p_copy.append(torch.tensor(p_copy, dtype=torch.float))
        all_copy_indices.append(torch.tensor(copy_indices))
        all_copy_values.append(copy_values)

    if corpus_type == "train":  # if corpus is train corpus generate vocabulary
        word_counter = OrderedCounter()
        for summary in summaries:
            word_counter.update(summary)
        for k in list(word_counter.keys()):
            if word_counter[k] < 2:
                del word_counter[k]  # will replace w/ unk
        vocab = Vocab(word_counter.keys(), eos_and_bos=True)
    elif path.exists(".cache/generator/vocab.pt"):  # else load vocab
        vocab = pickle.load(open(".cache/generator/vocab.pt", "rb"))
    else:  # if it doesn't exist create it
        _, _, _, _, vocab, _ = preproc_generator_data("train", extractor, planner, folder, dataset)

    summaries = pad_sequence([torch.tensor([vocab[word] for word in summary])
                             for summary in summaries], batch_first=True)
    all_copy_values = pad_sequence([torch.tensor([vocab[val] for val in seq])
                                   for seq in all_copy_values], batch_first=True)
    all_p_copy = pad_sequence(all_p_copy, batch_first=True)
    all_copy_indices = pad_sequence(all_copy_indices, batch_first=True)

    # wite stuff to disk
    if not path.exists(".cache/generator"):
        makedirs(".cache/generator")
    torch.save(summaries, f".cache/generator/{corpus_type}_summaries.pt")
    torch.save(all_p_copy, f".cache/generator/{corpus_type}_p_copy.pt")
    torch.save(all_copy_indices, f".cache/generator/{corpus_type}_copy_indices.pt")
    torch.save(all_copy_values, f".cache/generator/{corpus_type}_copy_values.pt")
    torch.save(content_plans, f".cache/generator/{corpus_type}_content_plans.pt")
    pickle.dump(vocab, open(".cache/generator/vocab.pt", "wb"))

    return summaries, all_p_copy, all_copy_indices, all_copy_values, content_plans, vocab


def load_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    """Load a dataset e.g. for use with a dataloader"""
    try:
        summaries = torch.load(f".cache/generator/{corpus_type}_summaries.pt")
        p_copy = torch.load(f".cache/generator/{corpus_type}_p_copy.pt")
        copy_indices = torch.load(f".cache/generator/{corpus_type}_copy_indices.pt")
        copy_values = torch.load(f".cache/generator/{corpus_type}_copy_values.pt")
        content_plans = torch.load(f".cache/generator/{corpus_type}_content_plans.pt")
        vocab = pickle.load(open(".cache/generator/vocab.pt", "rb"))

    except FileNotFoundError:
        logging.warning(f"Failed to locate cached generator {corpus_type} corpus!")
        logging.info(f"Genrating a new corpus...")
        summaries, p_copy, copy_indices, copy_values, content_plans, vocab = preproc_generator_data(
            corpus_type, extractor, planner, folder, dataset)

    idx2word = dict(((v, k) for k, v in vocab.items()))
    return CopyDataset(summaries, p_copy, copy_indices, copy_values, content_plans, vocab, idx2word)
