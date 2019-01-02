from json import loads
from collections import Counter, defaultdict
from math import floor
from random import shuffle
from nltk import sent_tokenize
from word2number import w2n
from os import path, makedirs
from abc import ABC, abstractmethod
import torch
import tarfile


class Loader(ABC):
    bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
               "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
               "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
               "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
               "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

    ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
               "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
               "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

    NUM_PLAYERS = 13

    def __init__(self, folder="boxscore-data", dataset="rotowire"):
        try:
            print("Attempting to load cached data...")
            data = self.load_cached_data()
            print("Success!")
        except FileNotFoundError:
            print("Failed to locate files")
            print("Generating data from datasets...")
            path = folder + "/" + dataset + ".tar.bz2"

            with tarfile.open(path, 'r:bz2') as f:
                trdata = loads(f.extractfile(dataset + "/train.json").read())
                valdata = loads(f.extractfile(dataset + "/valid.json").read())
                testdata = loads(f.extractfile(dataset + "/test.json").read())
                data = self.preproc_datasets(trdata, valdata, testdata)

        self.trdata = data[0]
        self.valdata = data[1]
        self.testdata = data[2]

    @abstractmethod
    def load_cached_data(self):
        pass

    @abstractmethod
    def preproc_datasets(self):
        pass


class ExtractorLoader(Loader):
    prons = set(["he", "He", "him", "Him", "his", "His", "they",
                 "They", "them", "Them", "their", "Their"])  # leave out "it"
    singular_prons = set(["he", "He", "him", "Him", "his", "His"])
    plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

    number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])

    def __init__(self, folder="boxscore-data", dataset="rotowire"):
        super(ExtractorLoader, self).__init__(folder, dataset)

    def load_cached_data(self):
        trsents = torch.load(".cache/extractor/trsents.pt")
        trlens = torch.load(".cache/extractor/trlens.pt")
        trentdists = torch.load(".cache/extractor/trentdists.pt")
        trnumdists = torch.load(".cache/extractor/trnumdists.pt")
        trlabels = torch.load(".cache/extractor/trlabels.pt")
        trdata = (trsents, trlens, trentdists, trnumdists, trlabels)

        valsents = torch.load(".cache/extractor/valsents.pt")
        vallens = torch.load(".cache/extractor/vallens.pt")
        valentdists = torch.load(".cache/extractor/valentdists.pt")
        valnumdists = torch.load(".cache/extractor/valnumdists.pt")
        vallabels = torch.load(".cache/extractor/vallabels.pt")
        valdata = (valsents, vallens, valentdists, valnumdists, vallabels)

        testsents = torch.load(".cache/extractor/testsents.pt")
        testlens = torch.load(".cache/extractor/testlens.pt")
        testentdists = torch.load(".cache/extractor/testentdists.pt")
        testnumdists = torch.load(".cache/extractor/testnumdists.pt")
        testlabels = torch.load(".cache/extractor/testlabels.pt")
        testdata = (testsents, testlens, testentdists, testnumdists, testlabels)

        return (trdata, valdata, testdata)

    def get_ents(self, dat):
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

    def deterministic_resolve(self, pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
        # we'll just take closest compatible one.
        # first look in current sentence; if there's an antecedent here return None, since
        # we'll catch it anyway
        for j in range(len(curr_ents) - 1, -1, -1):
            if pron in self.singular_prons and curr_ents[j][2] in players:
                return None
            elif pron in self.plural_prons and curr_ents[j][2] in teams:
                return None
            elif pron in self.plural_prons and curr_ents[j][2] in cities:
                return None

        # then look in previous max_back sentences
        if len(prev_ents) > 0:
            for i in range(len(prev_ents) - 1, len(prev_ents) - 1 - max_back, -1):
                for j in range(len(prev_ents[i]) - 1, -1, -1):
                    if pron in self.singular_prons and prev_ents[i][j][2] in players:
                        return prev_ents[i][j]
                    elif pron in self.plural_prons and prev_ents[i][j][2] in teams:
                        return prev_ents[i][j]
                    elif pron in self.plural_prons and prev_ents[i][j][2] in cities:
                        return prev_ents[i][j]
        return None

    def extract_entities(self, sent, all_ents, prons, prev_ents=None, resolve_prons=False,
                         players=None, teams=None, cities=None):
        sent_ents = []
        i = 0
        while i < len(sent):
            if sent[i] in prons:
                if resolve_prons:
                    referent = self.deterministic_resolve(
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

    def annoying_number_word(self, sent, i):
        ignores = set(["three point", "three - point",
                       "three - pt", "three pt"])
        return " ".join(sent[i:i + 3]) not in ignores and " ".join(sent[i:i + 2]) not in ignores

    def extract_numbers(self, sent):
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
            elif toke in self.number_words and not self.annoying_number_word(sent, i):
                j = 1
                while i + j <= len(sent) and sent[i + j] in self.number_words and not self.annoying_number_word(sent, i + j):
                    j += 1
                try:
                    sent_nums.append(
                        (i, i + j, w2n.word_to_num(" ".join(sent[i:i + j]))))
                except ValueError:
                    print(sent)
                    print(sent[i:i + j])
                    assert False
                i += j
            else:
                i += 1
        return sent_nums

    def get_player_idx(self, bs, entname):
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
        assert len(keys) <= 1, entname + " : " + \
            str(bs["PLAYER_NAME"].values())
        return keys[0] if len(keys) > 0 else None

    def get_rels(self, entry, ents, nums, players, teams, cities):
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
                pidx = self.get_player_idx(bs, entname)
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
                                # rels.append((ent, numtup, "TEAM-" + colname, is_home))
                                # apparently I appended TEAM- at some pt...
                                rels.append((ent, numtup, colname, is_home))
                                found = True
                    if not found:
                        # should i specialize the NONE labels too?
                        rels.append((ent, numtup, "NONE", None))
        return rels

    def append_candidate_rels(self, entry, summ, all_ents, prons, players, teams, cities, candrels):
        """
        appends tuples of form (sentence_tokens, [rels]) to candrels
        """
        sents = sent_tokenize(summ)
        for j, sent in enumerate(sents):
            tokes = sent.split()
            ents = self.extract_entities(tokes, all_ents, prons)
            nums = self.extract_numbers(tokes)
            rels = self.get_rels(entry, ents, nums, players, teams, cities)
            if len(rels) > 0:
                candrels.append((tokes, rels))
        return candrels

    def get_candidate_rels(self, trdata, valdata, testdata):
        all_ents, players, teams, cities = self.get_ents(trdata)

        extracted_stuff = []
        for dataset in (trdata, valdata, testdata):
            nugz = []
            for i, entry in enumerate(dataset):
                summ = " ".join(entry['summary'])
                self.append_candidate_rels(entry, summ, all_ents, self.prons, players, teams, cities, nugz)

            extracted_stuff.append(nugz)

        del all_ents
        del players
        del teams
        del cities
        return extracted_stuff

    def append_to_data(self, tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
        """
        tup is (sent, [rels]);
        each rel is ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
        """
        sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
        sentlen = len(sent)
        sent.extend([-1] * (max_len - sentlen))
        for rel in tup[1]:
            ent, num, label, idthing = rel
            sents.append(sent)
            lens.append(sentlen)
            ent_dists = [j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in range(max_len)]
            entdists.append(ent_dists)
            num_dists = [j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in range(max_len)]
            numdists.append(num_dists)
            labels.append(labeldict[label])

    def append_multilabeled_data(self, tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
        """
        used for val, since we have contradictory labelings...
        tup is (sent, [rels]);
        each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
        """
        sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
        sentlen = len(sent)
        sent.extend([-1] * (max_len - sentlen))
        # get all the labels for the same rel
        unique_rels = defaultdict(list)
        for rel in tup[1]:
            ent, num, label, idthing = rel
            unique_rels[ent, num].append(label)

        for rel, label_list in unique_rels.items():
            ent, num = rel
            sents.append(sent)
            lens.append(sentlen)
            ent_dists = [j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in range(max_len)]
            entdists.append(ent_dists)
            num_dists = [j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in range(max_len)]
            numdists.append(num_dists)
            labels.append([labeldict[label] for label in label_list])

    def append_labelnums(self, labels):
        labelnums = [len(labellist) for labellist in labels]
        max_num_labels = max(labelnums)
        print("Maximum number of labels:", max_num_labels)

        # append number of labels to labels
        for i, labellist in enumerate(labels):
            labellist.extend([-1] * (max_num_labels - len(labellist)))
            labellist.append(labelnums[i])

    def get_player_idxs(self, entry):
        nplayers = 0
        home_players, vis_players = [], []
        for k, v in entry["box_score"]["PTS"].items():
            nplayers += 1

        num_home, num_vis = 0, 0
        for i in range(nplayers):
            player_city = entry["box_score"]["TEAM_CITY"][str(i)]
            if player_city == entry["home_city"]:
                if len(home_players) < self.NUM_PLAYERS:
                    home_players.append(str(i))
                    num_home += 1
            else:
                if len(vis_players) < self.NUM_PLAYERS:
                    vis_players.append(str(i))
                    num_vis += 1
        return home_players, vis_players

    def preproc_datasets(self, trdata, valdata, testdata, multilabel_train=True, nonedenom=0):
        rel_datasets = self.get_candidate_rels(trdata, valdata, testdata)
        # make vocab and get labels
        word_counter = Counter()
        [word_counter.update(tup[0]) for tup in rel_datasets[0]]
        for k in list(word_counter.keys()):
            if word_counter[k] < 2:
                del word_counter[k]  # will replace w/ unk
        word_counter["UNK"] = 1
        vocab = dict(((wrd, i + 1) for i, wrd in enumerate(word_counter.keys())))
        labelset = set()
        [labelset.update([rel[2] for rel in tup[1]]) for tup in rel_datasets[0]]
        labeldict = dict(((label, i + 1) for i, label in enumerate(labelset)))

        # save stuff
        trsents, trlens, trentdists, trnumdists, trlabels = [], [], [], [], []
        valsents, vallens, valentdists, valnumdists, vallabels = [], [], [], [], []
        testsents, testlens, testentdists, testnumdists, testlabels = [], [], [], [], []

        max_trlen = max((len(tup[0]) for tup in rel_datasets[0]))
        print("Maximum training sentence length:", max_trlen)

        # do training data
        for tup in rel_datasets[0]:
            if multilabel_train:
                self.append_multilabeled_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)
            else:
                self.append_to_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)

        if multilabel_train:
            self.append_labelnums(trlabels)

        if nonedenom > 0:
            # don't keep all the NONE labeled things
            none_idxs = [i for i, labellist in enumerate(trlabels) if labellist[0] == labeldict["NONE"]]
            shuffle(none_idxs)
            # allow at most 1/(nonedenom+1) of NONE-labeled
            num_to_keep = int(floor(float(len(trlabels) - len(none_idxs)) / nonedenom))
            print("Originally", len(trlabels), "training examples")
            print("Keeping", num_to_keep, "NONE-labeled examples")
            ignore_idxs = set(none_idxs[num_to_keep:])

            # get rid of most of the NONE-labeled examples
            trsents = [thing for i, thing in enumerate(trsents) if i not in ignore_idxs]
            trlens = [thing for i, thing in enumerate(trlens) if i not in ignore_idxs]
            trentdists = [thing for i, thing in enumerate(trentdists) if i not in ignore_idxs]
            trnumdists = [thing for i, thing in enumerate(trnumdists) if i not in ignore_idxs]
            trlabels = [thing for i, thing in enumerate(trlabels) if i not in ignore_idxs]

        print(len(trsents), "training examples")

        # do val, which we also consider multilabel
        max_vallen = max((len(tup[0]) for tup in rel_datasets[1]))
        for tup in rel_datasets[1]:
            self.append_multilabeled_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_vallen)

        self.append_labelnums(vallabels)

        print(len(valsents), "validation examples")

        # do test, which we also consider multilabel
        max_testlen = max((len(tup[0]) for tup in rel_datasets[2]))
        for tup in rel_datasets[2]:
            self.append_multilabeled_data(tup, testsents, testlens, testentdists, testnumdists, testlabels, vocab, labeldict, max_testlen)

        self.append_labelnums(testlabels)

        print(len(testsents), "test examples")

        # create tensors and save to disk
        if not path.exists(".cache/extractor"):
            makedirs(".cache/extractor")

        trsents = torch.tensor(trsents)
        trlens = torch.tensor(trlens)
        trentdists = torch.tensor(trentdists)
        trnumdists = torch.tensor(trnumdists)
        trlabels = torch.tensor(trlabels)
        torch.save(trsents, ".cache/extractor/trsents.pt")
        torch.save(trlens, ".cache/extractor/trlens.pt")
        torch.save(trentdists, ".cache/extractor/trentdists.pt")
        torch.save(trnumdists, ".cache/extractor/trnumdists.pt")
        torch.save(trlabels, ".cache/extractor/trlabels.pt")
        trdata = (trsents, trlens, trentdists, trnumdists, trlabels)

        valsents = torch.tensor(valsents)
        vallens = torch.tensor(vallens)
        valentdists = torch.tensor(valentdists)
        valnumdists = torch.tensor(valnumdists)
        vallabels = torch.tensor(vallabels)
        torch.save(valsents, ".cache/extractor/valsents.pt")
        torch.save(vallens, ".cache/extractor/vallens.pt")
        torch.save(valentdists, ".cache/extractor/valentdists.pt")
        torch.save(valnumdists, ".cache/extractor/valnumdists.pt")
        torch.save(vallabels, ".cache/extractor/vallabels.pt")
        valdata = (valsents, vallens, valentdists, valnumdists, vallabels)

        testsents = torch.tensor(testsents)
        testlens = torch.tensor(testlens)
        testentdists = torch.tensor(testentdists)
        testnumdists = torch.tensor(testnumdists)
        testlabels = torch.tensor(testlabels)
        torch.save(testsents, ".cache/extractor/testsents.pt")
        torch.save(testlens, ".cache/extractor/testlens.pt")
        torch.save(testentdists, ".cache/extractor/testentdists.pt")
        torch.save(testnumdists, ".cache/extractor/testnumdists.pt")
        torch.save(testlabels, ".cache/extractor/testlabels.pt")
        testdata = (testsents, testlens, testentdists, testnumdists, testlabels)

        return (trdata, valdata, testdata)


# class SelectorLoader(Loader):
#     def __init__(self, folder="boxscore-data", dataset="rotowire"):
#         super(SelectorLoader, self).__init__(folder, dataset)
#
#     def handle_word(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1
#         return self.word2index[word]
#
#     def box_preproc2(self):
#         """
#         just gets src for now
#         """
#         srcs = [[] for i in range(2 * self.NUM_PLAYERS + 2)]
#
#         for entry in self.trdata:
#             home_players, vis_players = self.get_player_idxs(entry)
#             for ii, player_list in enumerate([home_players, vis_players]):
#                 for j in range(self.NUM_PLAYERS):
#                     src_j = []
#                     player_key = player_list[j] if j < len(
#                         player_list) else None
#                     for k, key in enumerate(self.bs_keys):
#                         rulkey = key.split('-')[1]
#                         val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
#                         src_j.append(val)
#                     srcs[ii * self.NUM_PLAYERS + j].append(src_j)
#
#             home_src, vis_src = [], []
#             # required for now, so that linescores can be processed together with boxscores
#             for k in range(len(self.bs_keys) - len(self.ls_keys)):
#                 home_src.append("PAD")
#                 vis_src.append("PAD")
#
#             for k, key in enumerate(self.ls_keys):
#                 home_src.append(entry["home_line"][key])
#                 vis_src.append(entry["vis_line"][key])
#
#             srcs[-2].append(home_src)
#             srcs[-1].append(vis_src)
#
#         return srcs
#
#     def preproc_datasets(self):
#         srcs = self.box_preproc2()
#         lsrcs = []
#         bs_len = len(self.bs_keys)
#         records = list()
#
#         # maps from a num-rows length list of lists of ntrain to an ntrain-length list of concatenated rows
#         for i in range(len(srcs[0])):
#             src_i = []
#             for j in range(len(srcs)):
#                 src_i.extend(srcs[j][i])
#             lsrcs.append(src_i)
#
#         for sample in lsrcs:
#             row = list()
#             for i in range(0, len(sample), bs_len):
#                 entity = sample[i]
#                 values = sample[i + 1:i + bs_len]
#
#                 if entity != "PAD":  # parse as boxscore data
#                     for value, type_ in zip(values, self.bs_keys[1:]):  # don't use PLAYER_NAME as value
#                         row.append((self.handle_word(type_), self.handle_word(entity), self.handle_word(value)))
#                 else:  # parse linescore data
#                     entity = sample[i + bs_len - 1]  # TEAM-NAME
#                     values = sample[i + bs_len - len(self.ls_keys):i + bs_len + 1]  # ignore the padded entries
#                     for value, type_ in zip(values, self.ls_keys):
#                         row.append((self.handle_word(type_), self.handle_word(entity), self.handle_word(value)))
#             records.append(row)
#
#         return records
#
#     def parse_summaries(self):
#         max_sent_len = 0
#         for entry in self.trdata:
#             summ = entry["summary"]
#             max_sent_len = len(summ) if len(summ) > max_sent_len else max_sent_len
#             for word in summ:
#                 self.handle_word(word)
#
#         return max_sent_len


loader = ExtractorLoader()
