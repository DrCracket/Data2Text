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
from os import path, makedirs
from json import loads
from .constants import device, TEXT_MAX_LENGTH, PAD_WORD, BOS_WORD, EOS_WORD, multi_word_cities, multi_word_teams
from .helper_funcs import extract_entities, extract_numbers, to_device, preproc_text
from .planner import load_planner_data
from .data_structures import OrderedCounter, Vocab, CopyDataset


def make_content_plan(planner, dataset):
    """
    Generate a content plan with a trained content planner for the generator.
    """
    data_dim1 = dataset.sequence.size(0)
    data_dim2 = dataset.sequence.size(1)
    # size = (#entries, records, hidden_size)
    content_plans = torch.zeros(data_dim1, data_dim2, dtype=torch.long, device=device)
    bos_tensor = torch.tensor([dataset.vocab[BOS_WORD]], device=device)
    planner.eval()
    planner.to(device)

    with torch.no_grad():
        for dim1 in range(len(dataset)):
            records, _ = to_device(dataset[dim1])
            hidden, cell = planner.init_hidden(records.unsqueeze(0))
            record_index = bos_tensor
            already_added = list()
            dim2 = 0
            for _ in range(data_dim2):
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
                if record_index == dataset.vocab[EOS_WORD]:
                    break
                # must be unique and shouldn't be unavailable
                if dataset.idx2word[records[record_index][0][2].item()] != "N/A" and record_index not in already_added:
                    content_plans[dim1][dim2] = record_index
                    already_added.append(record_index)
                    dim2 += 1

    return dataset.sequence, content_plans.cpu()


def make_train_content_plan(dataset):
    """
    Generate a content plan for the generator.
    Use the extractor to identify the records to copy.
    Only used for training.
    """
    data_dim1 = dataset.content_plan.size(0)
    data_dim2 = dataset.content_plan.where(dataset.content_plan == 0, torch.tensor([1])).sum(dim=1).max()
    # size = (#entries, records, hidden_size)
    content_plans = torch.zeros(data_dim1, data_dim2, dtype=torch.long, device=device)

    with torch.no_grad():
        for dim1 in range(len(dataset)):
            records, content_plan = to_device(dataset[dim1])
            content_plan_iterator = iter(content_plan)
            next(content_plan_iterator)  # skip BOS word

            for dim2, record_index in enumerate(content_plan_iterator):
                if record_index == dataset.vocab[EOS_WORD]:
                    # ugly workaround when the content plan is empty (only happens in one case)
                    # in this case add an unrelated record to avoid an empty content plan
                    if dim2 == 0:
                        index = torch.tensor([629], device=device)
                        content_plans[dim1][dim2] = index
                    break
                content_plans[dim1][dim2] = record_index

    return dataset.sequence, content_plans.cpu()


def extract_things_ordered(tokes, all_ents, records, entry_indices, idx2word):
    """
    Bring extracted things and ents in the order they appear in the text.
    Also extract the entity part of the tuples where the value refers to an entity.
    """
    ents = extract_entities(tokes, all_ents, list())
    nums = extract_numbers(tokes)

    # only copy an entity if number and entity combinations match a record
    true_ents = set()
    non_num_entities = set()
    for index in entry_indices:
        value = idx2word[records[index][2].item()]
        entity = idx2word[records[index][0].item()]
        for ent in ents:
            if ent[2] == value:
                for index2 in entry_indices:
                    value2 = idx2word[records[index2][2].item()]
                    entity2 = idx2word[records[index2][0].item()]
                    for num in nums:
                        if str(num[2]) == value2 and entity == entity2:
                            true_ents.add(ent)
                            non_num_entities.add(entity)
                            break

    true_ents.update(nums)
    for thing in true_ents:
        assert (thing[1] - thing[0]) == 1, "only things with length of 1 allowed"
    sorted_things = [thing[0:3] for thing in sorted(true_ents, key=lambda x: x[0]) if len(thing) == 3 or not thing[3]]

    return sorted_things, non_num_entities


def get_copy_probs(summary, entry_indices, records, vocab, idx2word):
    """
    Extract the probabilities of which words are copied from the dataset
    for one summary
    """
    all_ents = set()
    all_sents = list()
    all_p_copy = list()
    copy_indices = list()
    copy_values = list()
    next_expected = 0

    # get all entities and corresponding values from record indices
    for index in entry_indices:
        type_ = idx2word[records[index][1].item()]
        value = idx2word[records[index][2].item()]
        if type_ in ["PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME", "TEAM-NAME", "TEAM-CITY"]:
            all_ents.add(value)
        copy_values.append(value)

    for sent in sent_tokenize(" ".join(summary)):
        tokes = list()
        split_sent = preproc_text(sent)
        i = 0
        while i < len(split_sent):
            # reassemble multi-word cities/teams
            if " ".join(split_sent[i:i + 2]) in multi_word_cities + multi_word_teams:
                tokes.append(" ".join(split_sent[i:i + 2]))
                i += 2
            # sometimes name abbrevations don't contain dots in the dataset
            elif len(split_sent[i]) <= 4 and split_sent[i].replace(".", "") in all_ents:
                tokes.append(split_sent[i].replace(".", ""))
                i += 1
            else:
                tokes.append(split_sent[i])
                i += 1
        p_copy = [0] * len(tokes)
        ents_and_nums, non_num_entities = extract_things_ordered(tokes, all_ents, records, entry_indices, idx2word)
        already_added_ents = set()
        for thing in ents_and_nums:
            iter_list = list(enumerate(entry_indices))
            # continue with the next index after the last predicted
            iter_list = iter_list[next_expected:] + iter_list[:next_expected]
            if type(thing[2]) == int:
                # find the num in entites whose identifiers where already copied
                for i, index in iter_list:
                    entity = idx2word[records[index][0].item()]
                    value = idx2word[records[index][2].item()]
                    if entity in already_added_ents and str(thing[2]) == value:
                        p_copy[thing[0]:thing[1]] = [1]
                        copy_indices.append(i)
                        if i >= next_expected:
                            next_expected = i + 1
                        break
                # if they can't be found search in all entities that appear in the sentence
                else:
                    for i, index in iter_list:
                        entity = idx2word[records[index][0].item()]
                        value = idx2word[records[index][2].item()]
                        if entity in non_num_entities and str(thing[2]) == value:
                            p_copy[thing[0]:thing[1]] = [1]
                            copy_indices.append(i)
                            if i >= next_expected:
                                next_expected = i + 1
                            break
            else:
                for i, index in iter_list:
                    value = idx2word[records[index][2].item()]
                    entity = idx2word[records[index][0].item()]
                    if thing[2] == value and entity in non_num_entities:
                        p_copy[thing[0]:thing[1]] = [1]
                        copy_indices.append(i)
                        already_added_ents.add(entity)
                        if i >= next_expected:
                            next_expected = i + 1
                        break
        all_sents.extend(tokes)
        all_p_copy.extend(p_copy)

    return all_sents, all_p_copy, copy_indices, copy_values


def preproc_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    plan_dataset = load_planner_data(corpus_type, extractor, folder, dataset)
    if corpus_type == "train":
        records, content_plans = make_train_content_plan(plan_dataset)
    else:
        records, content_plans = make_content_plan(planner, plan_dataset)
    summaries = list()
    all_p_copy = list()
    all_copy_indices = list()
    all_copy_values = list()

    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())

    for idx, rel_idx in enumerate(plan_dataset.idx_list):
        entry_records = records[idx]
        entry_indices = content_plans[idx]
        summary = raw_dataset[rel_idx]["summary"]

        summary, p_copy, copy_indices, copy_values = get_copy_probs(summary, entry_indices[entry_indices > 0],
                                                                    entry_records, plan_dataset.vocab,
                                                                    plan_dataset.idx2word)
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
    elif path.exists("data/generator/vocab.pt"):  # else load vocab
        vocab = pickle.load(open("data/generator/vocab.pt", "rb"))
    else:  # if it doesn't exist create it
        _, _, _, _, vocab, _ = preproc_generator_data("train", extractor, planner, folder, dataset)

    summaries = pad_sequence([torch.tensor([vocab[word] for word in summary])
                             for summary in summaries], batch_first=True)
    all_copy_values = pad_sequence([torch.tensor([vocab[val] for val in seq])
                                   for seq in all_copy_values], batch_first=True)
    all_p_copy = pad_sequence(all_p_copy, batch_first=True)
    all_copy_indices = pad_sequence(all_copy_indices, batch_first=True)

    # wite stuff to disk
    if not path.exists("data/generator"):
        makedirs("data/generator")
    torch.save(summaries, f"data/generator/{corpus_type}_summaries.pt")
    torch.save(all_p_copy, f"data/generator/{corpus_type}_p_copy.pt")
    torch.save(all_copy_indices, f"data/generator/{corpus_type}_copy_indices.pt")
    torch.save(all_copy_values, f"data/generator/{corpus_type}_copy_values.pt")
    torch.save(records, f"data/generator/{corpus_type}_records.pt")
    torch.save(content_plans, f"data/generator/{corpus_type}_content_plans.pt")
    pickle.dump(vocab, open("data/generator/vocab.pt", "wb"))
    pickle.dump(plan_dataset.idx_list, open(f"data/generator/{corpus_type}_idx_list.pt", "wb"))

    return summaries, all_p_copy, all_copy_indices, all_copy_values, records, content_plans, vocab, plan_dataset.idx_list


def load_generator_data(corpus_type, extractor, planner, folder="boxscore-data", dataset="rotowire"):
    """
    Load a dataset e.g. for use with a dataloader
    """
    try:
        summaries = torch.load(f"data/generator/{corpus_type}_summaries.pt")
        p_copy = torch.load(f"data/generator/{corpus_type}_p_copy.pt")
        copy_indices = torch.load(f"data/generator/{corpus_type}_copy_indices.pt")
        copy_values = torch.load(f"data/generator/{corpus_type}_copy_values.pt")
        records = torch.load(f"data/generator/{corpus_type}_records.pt")
        content_plans = torch.load(f"data/generator/{corpus_type}_content_plans.pt")
        vocab = pickle.load(open("data/generator/vocab.pt", "rb"))
        idx_list = pickle.load(open(f"data/generator/{corpus_type}_idx_list.pt", "rb"))

    except FileNotFoundError:
        logging.warning(f"Failed to locate generator {corpus_type} corpus!")
        logging.info(f"Generating a new corpus...")
        summaries, p_copy, copy_indices, copy_values, records, content_plans, vocab, idx_list = preproc_generator_data(
            corpus_type, extractor, planner, folder, dataset)

    idx2word = dict(((v, k) for k, v in vocab.items()))
    return CopyDataset(summaries, p_copy, copy_indices, copy_values, records, content_plans, vocab, idx2word, idx_list)


class TextGeneratorWrapper():
    """
    a wrapper around the pytorch text generator module to make it easy to
    generate text
    """
    generator = None

    def __init__(self, generator):
        self.generator = generator.eval().to(device)

    def generate_text(self, vocab, idx2word, entry):
        _, _, records, content_plan, _, copy_values = entry

        records, content_plan, copy_values = to_device([records.unsqueeze(0),
                                                        content_plan.unsqueeze(0), copy_values.unsqueeze(0)])
        # remove all the zero padded values from the content plans
        content_plan = content_plan[:, :(content_plan > vocab[PAD_WORD]).sum(dim=1)]
        hidden, cell = self.generator.init_hidden(records, content_plan)

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
