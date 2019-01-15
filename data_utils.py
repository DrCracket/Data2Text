###############################################################################
# This file contains all the needed utilities to transform the dataset        #
# to the required format datasets for the Extraction, Content Selection &     #
# Planning and Text Generation Module.                                        #
###############################################################################

from json import loads
from collections import Counter, OrderedDict
from nltk import sent_tokenize
from word2number import w2n
from os import path, makedirs
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from Levenshtein import setratio
import torch
import tarfile
import pickle

###############################################################################
# Helper Data Structures                                                      #
###############################################################################


HOME = "HOME"
AWAY = "AWAY"

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'


class DefaultListOrderedDict(OrderedDict):
    def __missing__(self, k):
        self[k] = []
        return self[k]


class Vocab(dict):
    """"dict that uses the length of the dict as value for every mew key"""

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
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

###############################################################################
# Dataset Classes                                                             #
###############################################################################


class ExtractorDataset(Dataset):
    sents = None
    entdists = None
    numdists = None
    labels = None
    idx2word = None
    idx2type = None
    stats = None
    len_entries = None

    def __init__(self, sents, entdists, numdists, labels, idx2word, idx2type, stats, len_entries):
        super().__init__()
        self.sents = sents
        self.entdists = entdists
        self.numdists = numdists
        self.labels = labels
        self.idx2word = idx2word
        self.idx2type = idx2type
        self.stats = stats
        self.len_entries = len_entries

    def __getitem__(self, idx):
        return self.sents[idx], self.entdists[idx], self.numdists[idx], self.labels[idx]

    def __len__(self):
        return (len(self.sents))

    def split(self, idx):
        return zip(self.sents.split(idx), self.entdists.split(idx), self.numdists.split(idx), self.labels.split(idx))


class PlannerDataset(Dataset):
    records = None
    content_plan = None
    idx2word = None
    stats = None

    def __init__(self, records, content_plan, idx2word, stats):
        self.records = records
        self.content_plan = content_plan
        self.idx2word = idx2word
        self.stats = stats

    def __getitem__(self, idx):
        return (self.records[idx], self.content_plan[idx])

    def __len__(self):
        return (len(self.records))

###############################################################################
# Helper Functions                                                            #
# Most of them are based on code of the functions from Wiseman et al.         #
# https://github.com/harvardnlp/data2text/blob/master/data_utils.py           #
###############################################################################


prons = ["he", "He", "him", "Him", "his", "His", "they",
         "They", "them", "Them", "their", "Their"]  # leave out "it"
singular_prons = ["he", "He", "him", "Him", "his", "His"]
plural_prons = ["they", "They", "them", "Them", "their", "Their"]
number_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"]
bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
           "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
           "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
           "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
           "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
           "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
           "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

NUM_PLAYERS = 13

MAX_RECORDS = 2 * NUM_PLAYERS * len(bs_keys) + 2 * len(ls_keys)


def get_ents(dat):
    players = set()
    teams = set()
    cities = set()
    for thing in dat:
        teams.add(thing["vis_name"])
        teams.add(thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["vis_city"] + " " + thing["vis_name"])
        teams.add(thing["vis_city"] + " " + thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["home_name"])
        teams.add(thing["home_line"]["TEAM-NAME"])
        teams.add(thing["home_city"] + " " + thing["home_name"])
        teams.add(thing["home_city"] + " " + thing["home_line"]["TEAM-NAME"])
        # special case for this
        if thing["vis_city"] == "Los Angeles":
            teams.add("LA" + thing["vis_name"])
        if thing["home_city"] == "Los Angeles":
            teams.add("LA" + thing["home_name"])
        # sometimes team_city is different
        cities.add(thing["home_city"])
        cities.add(thing["vis_city"])
        players.update(thing["box_score"]["PLAYER_NAME"].values())
        cities.update(thing["box_score"]["TEAM_CITY"].values())

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        entset.add(piece)

    all_ents = players | teams | cities

    return all_ents, players, teams, cities


def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None, since
    # we'll catch it anyway
    for j in range(len(curr_ents) - 1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in range(len(prev_ents) - 1, len(prev_ents) - 1 - max_back, -1):
            for j in range(len(prev_ents[i]) - 1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
                     players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(
                    sent[i], players, teams, cities, sent_ents, prev_ents)
                if referent is None:
                    # is a pronoun
                    sent_ents.append((i, i + 1, sent[i], True))
                else:
                    # pretend it's not a pron and put in matching string
                    sent_ents.append((i, i + 1, referent[2], False))
            else:
                sent_ents.append((i, i + 1, sent[i], True))  # is a pronoun
            i += 1
        # findest longest spans; only works if we put in words...
        elif sent[i] in all_ents:
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            sent_ents.append(
                (i, i + j - 1, " ".join(sent[i:i + j - 1]), False))
            i += j - 1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = ["three point", "three - point", "three - pt", "three pt", "three - pointers", "three - pointer", "three pointers"]
    return " ".join(sent[i:i + 3]) in ignores or " ".join(sent[i:i + 2]) in ignores


def extract_numbers(sent):
    sent_nums = []
    i = 0
    while i < len(sent):
        toke = sent[i]
        a_number = False
        try:
            int(toke)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append((i, i + 1, int(toke)))
            i += 1
        # get longest span  (this is kind of stupid)
        elif toke in number_words and not annoying_number_word(sent, i):
            j = 1
            while i + j < len(sent) and sent[i + j] in number_words and not annoying_number_word(sent, i + j):
                j += 1
            sent_nums.append((i, i + j, w2n.word_to_num(" ".join(sent[i:i + j]))))
            i += j
        else:
            i += 1
    return sent_nums


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].items():
        if entname == v:
            keys.append(k)
    if len(keys) == 0:
        for k, v in bs["SECOND_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # take the earliest one
            keys.sort(key=lambda x: int(x))
            keys = keys[:1]
    if len(keys) == 0:
        for k, v in bs["FIRST_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # if we matched on first name and there are a bunch just forget about it
            return None
    assert len(keys) <= 1, entname + " : " + str(bs["PLAYER_NAME"].values())
    return keys[0] if len(keys) > 0 else None


def get_rels(entry, ents, nums, players, teams, cities):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    for i, ent in enumerate(ents):
        if ent[3]:  # pronoun
            continue  # for now
        entname = ent[2]
        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if pidx is not None:  # player might not actually be in the game or whatever
                    for colname, col in bs.items():
                        if col[pidx] == strnum:  # allow multiple for now
                            rels.append(
                                (ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None))

        else:  # has to be city or team
            entpieces = entname.split()
            linescore = None
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    linescore = entry["home_line"]
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    linescore = entry["vis_line"]
                    is_home = False
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if linescore is not None:
                    for colname, val in linescore.items():
                        if val == strnum:
                            rels.append((ent, numtup, colname, is_home))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None))
    return rels


def append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities):
    """
    Appends tuples of form (sentence_tokens, [rels]) to candrels.
    """
    sents = sent_tokenize(summ)
    candrels = []
    for j, sent in enumerate(sents):
        tokes = sent.split()
        ents = extract_entities(tokes, all_ents, prons)
        nums = extract_numbers(tokes)
        rels = get_rels(entry, ents, nums, players, teams, cities)
        if len(rels) > 0:
            candrels.append((tokes, rels))
    return candrels


def get_candidate_rels(dataset):
    all_ents, players, teams, cities = get_ents(dataset["train"])

    extracted_stuff = {}
    for corpus_type, entries in dataset.items():
        candrels = []
        for i, entry in enumerate(entries):
            summ = " ".join(entry['summary'])
            candrels.append(append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities))

        extracted_stuff[corpus_type] = candrels

    del all_ents
    del players
    del teams
    del cities
    return extracted_stuff


def append_multilabeled_data(entry, sents, entdists, numdists, labels, vocab, labeldict, max_len, len_entries):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    len_entry holds the information how many tups belong to one entry.
    This is later used for the creation of dataset for the content planner.
    """
    len_entry = 0
    for tup in entry:
        sent = [vocab[wrd] for wrd in tup[0]]
        sentlen = len(sent)
        sent.extend([0] * (max_len - sentlen))
        # get all the labels for the same rel
        unique_rels = DefaultListOrderedDict()
        for rel in tup[1]:
            ent, num, label, idthing = rel
            unique_rels[ent, num].append(label)

        for rel, label_list in unique_rels.items():
            ent, num = rel
            len_entry += 1
            sents.append(sent)
            ent_dists = [j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in range(max_len)]
            entdists.append(ent_dists)
            num_dists = [j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in range(max_len)]
            numdists.append(num_dists)
            labels.append([labeldict[label] for label in label_list])
    if len_entry != 0:
        len_entries.append(len_entry)


def get_player_idxs(entry):
    nplayers = 0
    home_players, vis_players = [], []
    for k, v in entry["box_score"]["PTS"].items():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in range(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1
    return home_players, vis_players


def preproc_extractor_data(set_type, folder, dataset_name, train_stats=None):
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

    print(f"Generating {set_type} examples from {location}...")
    sents, entdists, numdists, labels, len_entries = [], [], [], [], []
    max_len = max((len(tup[0]) for entry in extracted_rel_tups[set_type] for tup in entry))
    for entry in extracted_rel_tups[set_type]:
        append_multilabeled_data(entry, sents, entdists, numdists, labels, vocab, labeldict, max_len, len_entries)
    print(f"Generated {len(sents)} {set_type} examples!")

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

    if set_type != "train":
        # load the train stats, that are used to remove unseen data from train and test set
        if path.exists(".cache/extractor/stats.pt"):
            # if they are not cached, create them
            train_stats = pickle.load(open(".cache/extractor/stats.pt", "rb"))
        else:
            _, _, _, train_stats, _ = preproc_extractor_data("train", folder, dataset_name)

        # clamp values so no other distances except the ones in the train set occur
        entdists_ts = entdists_ts.clamp(train_stats["min_entd"], train_stats["max_entd"])
        numdists_ts = numdists_ts.clamp(train_stats["min_numd"], train_stats["max_numd"])
    else:  # the current set_type is "train" so we need to create the train_stats
        min_entd, max_entd = entdists_ts.min().item(), entdists_ts.max().item()
        min_numd, max_numd = numdists_ts.min().item(), numdists_ts.max().item()

        # save shifted number to shift back when a content plan is generated
        entshift = min_entd
        numshift = min_numd

        # the number of entries the distance embedding has to have
        max_dist = max(max_entd, max_numd) - min(min_entd, min_numd) + 1

        n_words = len(vocab)
        n_types = len(labeldict)

        train_stats = {"entshift": entshift, "numshift": numshift, "max_dist": max_dist,
                       "min_entd": min_entd, "max_entd": max_entd, "min_numd": min_numd,
                       "max_numd": max_numd, "n_words": n_words, "n_types": n_types}
        pickle.dump(train_stats, open(".cache/extractor/stats.pt", "wb"))

    # shift values to eliminate negative numbers
    entdists_ts -= train_stats["min_entd"]
    numdists_ts -= train_stats["min_numd"]

    # write dicts, lists and tensors to disk
    idx2word = dict(((v, k) for k, v in vocab.items()))
    idx2type = dict(((v, k) for k, v in labeldict.items()))
    torch.save(sents_ts, f".cache/extractor/{set_type}_sents.pt")
    torch.save(entdists_ts, f".cache/extractor/{set_type}_entdists.pt")
    torch.save(numdists_ts, f".cache/extractor/{set_type}_numdists.pt")
    torch.save(labels_ts, f".cache/extractor/{set_type}_labels.pt")
    pickle.dump(len_entries, open(f".cache/extractor/{set_type}_len_entries.pt", "wb"))
    pickle.dump(idx2word, open(".cache/extractor/vocab.pt", "wb"))
    pickle.dump(idx2type, open(".cache/extractor/labels.pt", "wb"))

    return (sents_ts, entdists_ts, numdists_ts, labels_ts), idx2word, idx2type, train_stats, len_entries


def add_special_records(records, idx):
    """special records for enabling pointing to bos and eos in first stage"""
    record = []
    record.append(BOS_WORD)
    record.append(PAD_WORD)
    record.append(PAD_WORD)
    record.append(PAD_WORD)
    records["SPECIAL_RECORDS"].append((idx, record))
    idx += 1
    record = []
    record.append(EOS_WORD)
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
        record.append(team)
        record.append(key)
        record.append(entry["vis_line"][key])
        record.append(AWAY)
        records[f"{city} {team}"].append((idx, record))
        idx += 1
        vocab.update(record) if vocab is not None else None

    return records, vocab


def preproc_planner_data(corpus_type, extractor, folder="boxscore-data", dataset="rotowire"):
    print(f"Processing records and content plans from the {corpus_type} corpus...")
    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())

    extr_dataset = load_extractor_data(corpus_type, folder, dataset)
    pre_content_plans = extractor.extract_relations(extr_dataset)
    # add two to MAX_RECORDS for BOS and EOS records
    records = torch.zeros(len(raw_dataset), MAX_RECORDS + 2, 4, dtype=torch.long)
    content_plans = torch.zeros(len(raw_dataset), MAX_RECORDS + 2, dtype=torch.long)
    stats = dict()

    if corpus_type == "train":  # if corpus is train corpus generate vocabulary
        vocab = Vocab(eos_and_bos=True)
    elif path.exists(".cache/planner/vocab.pt"):  # else load vocab
        vocab = pickle.load(open(".cache/planner/vocab.pt", "rb"))
    else:  # if it doesn't exist create it
        _, _, vocab, _ = preproc_planner_data("train", extractor, folder, dataset)

    # used by the planner to identify indices of special words
    stats["BOS_INDEX"] = torch.tensor([vocab[BOS_WORD]])
    stats["EOS_INDEX"] = torch.tensor([vocab[EOS_WORD]])
    stats["PAD_INDEX"] = torch.tensor([vocab[PAD_WORD]])

    # create the folder for the cached files if it does not exist
    if not path.exists(".cache/planner"):
        makedirs(".cache/planner")

    for dim1, (raw_entry, pre_content_plan) in enumerate(zip(raw_dataset, pre_content_plans)):
        # create the records from the dataset
        if corpus_type == "train":  # only update vocab if processed corpus is train corpus
            entry_records, vocab = create_records(raw_entry, vocab)
        else:
            entry_records, _ = create_records(raw_entry)
        content_plan = [vocab[BOS_WORD]]  # begin sequence with BOS word
        # for every extracted record in the ie content plan:
        for extr_record in pre_content_plan:
            # get the entity that has the highest string similarity (if the type of the extracted relation isn't NONE)
            entity = extr_record[0]
            value = extr_record[1]
            type_ = extr_record[2]
            # NONE-types indicate no relation and shouldn't be used in the content plan, unknown (UNK_WORD) values should be excluded as well
            if type_ != "NONE" and value != UNK_WORD:
                matched_entity = max(((key, setratio(key.split(), entity.split())) for key in entry_records.keys()), key=lambda word: word[1])
                # and if the similarity is reasonable (rule of thumb above 0.5) compare value and type of all records with that entity
                if matched_entity[1] >= 0.5:
                    matched_records = entry_records[matched_entity[0]]
                    for idx, record in matched_records:
                        # if type and values match a record exists and can be used in the content plan for the planning module
                        if type_ == record[1] and (value == record[2] or str(w2n.word_to_num(value)) == record[2]):
                            content_plan.append(idx)
                            break
        content_plan.append(vocab[EOS_WORD])  # end sequence with EOS word
        # translate words to indeces and create tensors
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
    pickle.dump(stats, open(".cache/planner/stats.pt", "wb"))

    return records, content_plans, vocab, stats

###############################################################################
# Functions to create datasets                                                #
###############################################################################


def load_extractor_data(corpus_type, folder="boxscore-data", dataset="rotowire"):
    try:
        sents = torch.load(f".cache/extractor/{corpus_type}_sents.pt")
        entdists = torch.load(f".cache/extractor/{corpus_type}_entdists.pt")
        numdists = torch.load(f".cache/extractor/{corpus_type}_numdists.pt")
        labels = torch.load(f".cache/extractor/{corpus_type}_labels.pt")
        len_entries = pickle.load(open(f".cache/extractor/{corpus_type}_len_entries.pt", "rb"))
        idx2word = pickle.load(open(".cache/extractor/vocab.pt", "rb"))
        idx2type = pickle.load(open(".cache/extractor/labels.pt", "rb"))
        stats = pickle.load(open(".cache/extractor/stats.pt", "rb"))

    except FileNotFoundError:
        print(f"Failed to locate cached {corpus_type} corpus!")
        type_data, idx2word, idx2type, stats, len_entries = preproc_extractor_data(corpus_type, folder, dataset)
        sents, entdists, numdists, labels = type_data

    return ExtractorDataset(sents, entdists, numdists, labels, idx2word, idx2type, stats, len_entries)


def load_planner_data(corpus_type, extractor, folder="boxscore-data", dataset="rotowire"):
    try:
        records = torch.load(f".cache/planner/{corpus_type}_records.pt")
        content_plans = torch.load(f".cache/planner/{corpus_type}_content_plans.pt")
        vocab = pickle.load(open(".cache/planner/vocab.pt", "rb"))
        stats = pickle.load(open(".cache/planner/stats.pt", "rb"))

    except FileNotFoundError:
        print(f"Failed to locate cached {corpus_type} corpus!")
        records, content_plans, vocab, stats = preproc_planner_data(corpus_type, extractor, folder, dataset)

    idx2word = dict(((v, k) for k, v in vocab.items()))
    return PlannerDataset(records, content_plans, idx2word, stats)
