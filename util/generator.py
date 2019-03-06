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
from .constants import number_words, suffixes, device, TEXT_MAX_LENGTH
from .helper_funcs import annoying_number_word, extract_entities, extract_numbers, to_device
from .planner import load_planner_data
from .constants import PAD_WORD, BOS_WORD, EOS_WORD, MAX_CONTENT_PLAN_LENGTH, MIN_CONTENT_PLAN_LENGTH
from .data_structures import OrderedCounter, Vocab, CopyDataset


def make_content_plan(planner, dataset):
    """
    Generate a content plan with a trained content planner for the generator.
    """
    dim1, _, dim3 = dataset.sequence.size(0), dataset.sequence.size(1), planner.hidden_size
    # size = (#entries, records, hidden_size)
    content_plans = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dim3, device=device)
    record_indices = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dtype=torch.long, device=device)
    bos_tensor = torch.tensor([dataset.vocab[BOS_WORD]], device=device)
    planner.eval()
    planner.to(device)

    with torch.no_grad():
        for dim1 in range(len(dataset)):
            records, _ = to_device(dataset[dim1])
            hidden, cell = planner.init_hidden(records.unsqueeze(0))
            record_index = bos_tensor
            dim2 = 0
            iteration = 0
            # a content plan has to include at least MIN_CONTENT_PLAN_LENGTH records
            while not record_index == dataset.vocab[EOS_WORD] or dim2 < MIN_CONTENT_PLAN_LENGTH:
                output, hidden, cell = planner(record_index, hidden, cell)
                # in 0.002% of all cases the content plan would be empty. To prevent that use the next likeliest
                # record in the distribution that isn't a special record like BOS, EOS, PAD
                if dim2 == 0:
                    _, top = torch.topk(output, 5, dim=1)
                    for index in top[0]:
                        if index > 4:
                            record_index = index.unsqueeze(0)
                            break
                else:
                    record_index = output.argmax(dim=1)
                # must be a number word and unique
                if dataset.idx2word[records[record_index][0][2].item()].isdigit() and record_index not in record_indices[dim1]:
                    # size = (1) => size = (1, 1, hidden_size)
                    idx = record_index.view(-1, 1, 1).repeat(1, 1, planner.hidden_size)
                    content_plans[dim1][dim2] = planner.selected_content.gather(1, idx)
                    record_indices[dim1][dim2] = record_index
                    dim2 += 1
                # allow at most MAX_CONTENT_PLAN_LENGTH sentences
                if iteration < MAX_CONTENT_PLAN_LENGTH - 1:
                    iteration += 1
                else:
                    break

    return content_plans.cpu(), record_indices.cpu()


def make_train_content_plan(planner, dataset):
    """
    Generate a content plan with a trained content planner for the generator.
    Use the extractor to identify the records to copy.
    Only used for training.
    """
    dim1, _, dim3 = dataset.sequence.size(0), dataset.sequence.size(1), planner.hidden_size
    # size = (#entries, records, hidden_size)
    content_plans = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dim3, device=device)
    record_indices = torch.zeros(dim1, MAX_CONTENT_PLAN_LENGTH, dtype=torch.long, device=device)
    planner.eval()
    planner.to(device)

    with torch.no_grad():
        for dim1 in range(len(dataset)):
            records, content_plan = to_device(dataset[dim1])
            planner.init_hidden(records.unsqueeze(0))
            content_plan_iterator = iter(content_plan)
            next(content_plan_iterator)  # skip BOS word

            for dim2, record_index in enumerate(content_plan_iterator):
                if record_index == dataset.vocab[EOS_WORD]:
                    break
                # size = (1) => size = (1, 1, hidden_size)
                idx = record_index.view(-1, 1, 1).repeat(1, 1, planner.hidden_size)
                content_plans[dim1][dim2] = planner.selected_content.gather(1, idx)
                record_indices[dim1][dim2] = record_index
                dim2 += 1

    return content_plans.cpu(), record_indices.cpu()


def get_copy_probs(summary, entry_indices, records, vocab, idx2word):
    """
    Extract the probabilities of which words are copied from the dataset
    """
    all_ents = set()
    all_sents = list()
    all_p_copy = list()
    copy_indices = list()
    copy_values = list()

    # get all entities and corresponding values from record indices
    for index in entry_indices:
        if index == vocab[PAD_WORD]:
            break
        all_ents.add(idx2word[records[index][0].item()])
        copy_values.append(idx2word[records[index][2].item()])

    # add stuff like first name and last name to the set
    entset = set()
    for entity in all_ents:
        for piece in entity.split(" "):
            if len(piece) > 1 and piece not in suffixes:
                entset.add(piece)
    all_ents = all_ents | entset

    for sent in sent_tokenize(" ".join(summary)):
        tokes = list()
        split_sent = sent.split(" ")
        for i, word in enumerate(split_sent):
            # replace every number word with the corresponding digits
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
                        if len(piece) > 1 and piece not in suffixes:
                            identifiers.add(piece)
                    if ent[2] in identifiers and str(num[2]) == value:
                        p_copy[num[0]:num[1]] = [1]
                        copy_indices.append(i)
                        break
        all_sents.extend(tokes)
        all_p_copy.extend(p_copy)

    return all_sents, all_p_copy, copy_indices, copy_values


def preproc_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    plan_dataset = load_planner_data(corpus_type, extractor, folder, dataset)
    if corpus_type == "train":
        content_plans, record_indices = make_train_content_plan(planner, plan_dataset)
    else:
        content_plans, record_indices = make_content_plan(planner, plan_dataset)
    summaries = list()
    all_p_copy = list()
    all_copy_indices = list()
    all_copy_values = list()

    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())

    for idx, rel_idx in enumerate(plan_dataset.idx_list):
        records, _ = plan_dataset[idx]
        entry_indices = record_indices[idx]
        summary = raw_dataset[rel_idx]["summary"]

        summary, p_copy, copy_indices, copy_values = get_copy_probs(summary, entry_indices, records,
                                                                    plan_dataset.vocab, plan_dataset.idx2word)
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
    pickle.dump(plan_dataset.idx_list, open(f".cache/generator/{corpus_type}_idx_list.pt", "wb"))

    return summaries, all_p_copy, all_copy_indices, all_copy_values, content_plans, vocab, plan_dataset.idx_list


def load_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    """
    Load a dataset e.g. for use with a dataloader
    """
    try:
        summaries = torch.load(f".cache/generator/{corpus_type}_summaries.pt")
        p_copy = torch.load(f".cache/generator/{corpus_type}_p_copy.pt")
        copy_indices = torch.load(f".cache/generator/{corpus_type}_copy_indices.pt")
        copy_values = torch.load(f".cache/generator/{corpus_type}_copy_values.pt")
        content_plans = torch.load(f".cache/generator/{corpus_type}_content_plans.pt")
        vocab = pickle.load(open(".cache/generator/vocab.pt", "rb"))
        idx_list = pickle.load(open(f".cache/generator/{corpus_type}_idx_list.pt", "rb"))

    except FileNotFoundError:
        logging.warning(f"Failed to locate cached generator {corpus_type} corpus!")
        logging.info(f"Genrating a new corpus...")
        summaries, p_copy, copy_indices, copy_values, content_plans, vocab, idx_list = preproc_generator_data(
            corpus_type, extractor, planner, folder, dataset)

    idx2word = dict(((v, k) for k, v in vocab.items()))
    return CopyDataset(summaries, p_copy, copy_indices, copy_values, content_plans, vocab, idx2word, idx_list)


class TextGeneratorWrapper():
    """
    a wrapper around the pytorch text generator module to make it easy to
    generate text
    """
    generator = None

    def __init__(self, generator):
        self.generator = generator.eval().to(device)

    def generate_text(self, vocab, idx2word, entry):
        _, _, content_plan, _, copy_values = entry

        content_plan, copy_values = to_device([content_plan.unsqueeze(0), copy_values.unsqueeze(0)])
        # remove all the zero padded values from the content plans
        non_zero = content_plan.nonzero()[:, 1].unique(sorted=True)
        non_zero = non_zero.view(1, -1, 1).repeat(1, 1, content_plan.size(2))
        hidden, cell = self.generator.init_hidden(content_plan.gather(1, non_zero))

        input_word = torch.tensor([vocab[BOS_WORD]], device=device)
        text = []

        with torch.no_grad():
            while input_word.cpu() != vocab[EOS_WORD] and len(text) <= TEXT_MAX_LENGTH:
                out_prob, copy_prob, p_copy, hidden, cell = self.generator(
                    input_word, hidden, cell)
                if p_copy > 0.5:
                    input_word = copy_values[:, copy_prob.argmax(dim=1)].view(1)
                else:
                    input_word = out_prob.argmax(dim=1)
                text.append((p_copy > 0.5, input_word.item()))

        # copied values are marked with bold markdown syntax
        markup = ["**" + idx2word[idx] + "**" if p_copy else idx2word[idx] for p_copy, idx in text[:-1]]
        normal = [idx2word[idx] for _, idx in text[:-1]]

        return markup, normal
