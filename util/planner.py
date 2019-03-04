###############################################################################
# Provides functions to load and create datasets for the                      #
# content planning module                                                     #
###############################################################################

import tarfile
import pickle
import torch
import logging
from torch.nn.utils.rnn import pad_sequence
from word2number import w2n
from os import path, makedirs
from json import loads
from .constants import PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, NUM_PLAYERS, bs_keys, ls_keys, device, HOME, AWAY, MAX_RECORDS
from .data_structures import Vocab, DefaultListOrderedDict, SequenceDataset
from .helper_funcs import get_player_idxs, to_device
from .extractor import load_extractor_data


def extract_relations(extractor, dataset):
    """Use a trained extractor to extract relations for the content planner"""
    total_relations = []
    extractor.eval()
    extractor.to(device)

    with torch.no_grad():
        for idx, (sents, entdists, numdists, _) in zip(dataset.idx_list, dataset.split(dataset.len_entries)):
            sents, entdists, numdists = to_device([sents, entdists, numdists])
            predictions = extractor.forward(sents, entdists, numdists)
            relations = []
            for prediction, sent, entdist, numdist in zip(predictions, sents, entdists, numdists):
                type_ = dataset.idx2type[prediction.argmax().item()]
                entity = []
                number = []
                for word, ent, num in zip(sent, entdist, numdist):
                    if ent.item() + dataset.stats["entshift"] == 0:
                        entity.append(dataset.idx2word[word.item()])
                    if num.item() + dataset.stats["numshift"] == 0:
                        number.append(dataset.idx2word[word.item()])
                relations.append([" ".join(entity), " ".join(number), type_])
            total_relations.append((idx, relations))
    return total_relations


def add_special_records(records, idx):
    """special records for enabling pointing to bos and eos in first stage"""

    # add two Padding Records, so that record indices match with vocab indices
    for entity in (PAD_WORD, PAD_WORD, BOS_WORD, EOS_WORD):
        record = []
        record.append(entity)
        record.append(PAD_WORD)
        record.append(PAD_WORD)
        record.append(PAD_WORD)
        records["SPECIAL_RECORDS"].append((idx, record))
        idx += 1

    return records, idx


def create_records(entry, vocab=None):
    idx = 0
    records = DefaultListOrderedDict()
    records, idx = add_special_records(records, idx)

    home_players, vis_players = get_player_idxs(entry)
    for ii, player_list in enumerate([home_players, vis_players]):
        for j in range(NUM_PLAYERS):
            player_key = player_list[j] if j < len(player_list) else None
            player_name = entry["box_score"]['PLAYER_NAME'][player_key] if player_key is not None else "N/A"
            for k, key in enumerate(bs_keys):
                rulkey = key.split('-')[1]
                val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
                record = list()
                record.append(player_name)
                record.append("PLAYER-" + rulkey)
                record.append(val)
                record.append(HOME if ii == 0 else AWAY)
                records[player_name].append((idx, record))
                idx += 1
                vocab.update(record) if vocab is not None else None

    for k, key in enumerate(ls_keys):
        record = list()
        team = entry["home_line"]["TEAM-NAME"]
        city = entry["home_line"]["TEAM-CITY"]
        if city == "Los Angeles":
            city = city + " LA"
        record.append(team)
        record.append(key)
        record.append(entry["home_line"][key])
        record.append(HOME)
        # use both city and team as keys as the are often used as synonyms
        records[f"{city} {team}"].append((idx, record))
        idx += 1
        vocab.update(record) if vocab is not None else None
    for k, key in enumerate(ls_keys):
        record = list()
        team = entry["vis_line"]["TEAM-NAME"]
        city = entry["vis_line"]["TEAM-CITY"]
        if city == "Los Angeles":
            city = city + " LA"
        record.append(team)
        record.append(key)
        record.append(entry["vis_line"][key])
        record.append(AWAY)
        records[f"{city} {team}"].append((idx, record))
        idx += 1
        vocab.update(record) if vocab is not None else None

    return records, vocab


def preproc_planner_data(corpus_type, extractor, folder="boxscore-data", dataset="rotowire"):
    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())

    extr_dataset = load_extractor_data(corpus_type, folder, dataset)
    pre_content_plans = extract_relations(extractor, extr_dataset)
    # add four to MAX_RECORDS for BOS, EOS and two initial PAD records
    records = torch.zeros(len(pre_content_plans), MAX_RECORDS + 4, 4, dtype=torch.long)
    content_plans = torch.zeros(len(pre_content_plans), MAX_RECORDS + 4, dtype=torch.long)

    if corpus_type == "train":  # if corpus is train corpus generate vocabulary
        vocab = Vocab(eos_and_bos=True)
    elif path.exists(".cache/planner/vocab.pt"):  # else load vocab
        vocab = pickle.load(open(".cache/planner/vocab.pt", "rb"))
    else:  # if it doesn't exist create it
        _, _, vocab, _, _ = preproc_planner_data("train", extractor, folder, dataset)

    # create the folder for the cached files if it does not exist
    if not path.exists(".cache/planner"):
        makedirs(".cache/planner")

    for dim1, (entry_idx, pre_content_plan) in enumerate(pre_content_plans):
        raw_entry = raw_dataset[entry_idx]  # load the entry that corresponds to a content plan
        if corpus_type == "train":  # only update vocab if processed corpus is train corpus
            entry_records, vocab = create_records(raw_entry, vocab)
        else:
            entry_records, _ = create_records(raw_entry)
        content_plan = [vocab[BOS_WORD]]  # begin sequence with BOS record
        # for every extracted record in the ie content plan:
        for extr_record in pre_content_plan:
            # get the entity that has the highest string similarity (if the type of the extracted relation isn't NONE)
            entity = extr_record[0]
            value = extr_record[1]
            type_ = extr_record[2]
            # NONE-types indicate no relation and shouldn't be used in the content plan,
            # unknown (UNK_WORD) values should be excluded as well
            if type_ != "NONE" and value != UNK_WORD:
                matched_entity = max(((key, len(set(key.split()).intersection(entity.split()))) for key in entry_records.keys()),
                                     key=lambda word: word[1])
                # and if the similarity is reasonable (if at least one word e.g. surname match)
                # compare value and type of all records with that entity
                if matched_entity[1] >= 1:
                    matched_records = entry_records[matched_entity[0]]
                    for idx, record in matched_records:
                        # if type and values match a record exists and can be
                        # used in the content plan for the planning module
                        if type_ == record[1] and (value == record[2] or str(w2n.word_to_num(value)) == record[2]):
                            content_plan.append(idx)
                            break
        content_plan.append(vocab[EOS_WORD])  # end sequence with EOS word
        # translate words to indices and create tensors
        for entity_records in entry_records.values():
            for dim2, record in entity_records:
                for dim3, word in enumerate(record):
                    records[dim1][dim2][dim3] = vocab[word]
        for dim2, record_idx in enumerate(content_plan):
            content_plans[dim1][dim2] = record_idx
    # pad lists of tensors to tensor of equal length
    records = pad_sequence(records, batch_first=True)
    content_plans = pad_sequence(content_plans, batch_first=True)
    # wite stuff to disk
    torch.save(records, f".cache/planner/{corpus_type}_records.pt")
    torch.save(content_plans, f".cache/planner/{corpus_type}_content_plans.pt")
    pickle.dump(vocab, open(".cache/planner/vocab.pt", "wb"))
    pickle.dump(extr_dataset.idx_list, open(f".cache/planner/{corpus_type}_idx_list.pt", "wb"))

    return records, content_plans, vocab, extr_dataset.idx_list


def load_planner_data(corpus_type, extractor, folder="boxscore-data", dataset="rotowire"):
    """Load a dataset e.g. for use with a dataloader"""
    try:
        records = torch.load(f".cache/planner/{corpus_type}_records.pt")
        content_plans = torch.load(f".cache/planner/{corpus_type}_content_plans.pt")
        vocab = pickle.load(open(".cache/planner/vocab.pt", "rb"))
        idx_list = pickle.load(open(f".cache/planner/{corpus_type}_idx_list.pt", "rb"))

    except FileNotFoundError:
        logging.warning(f"Failed to locate cached content planner {corpus_type} corpus!")
        logging.info(f"Genrating a new corpus...")
        records, content_plans, vocab, idx_list = preproc_planner_data(corpus_type, extractor, folder, dataset)

    idx2word = dict(((v, k) for k, v in vocab.items()))
    return SequenceDataset(records, content_plans, vocab, idx2word, idx_list)
