###############################################################################
# Helper Functions used for data preprocessing                                #
# They are based on the code from Wiseman et al.                              #
# https://github.com/harvardnlp/data2text/blob/master/data_utils.py           #
###############################################################################

from word2number import w2n
from nltk import sent_tokenize
from .constants import singular_prons, plural_prons, number_words, prons, device, suffixes, abbr2ent, NUM_PLAYERS
from .data_structures import DefaultListOrderedDict


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
        if thing["vis_city"] == "LA":
            teams.add("Los Angeles" + " " + thing["vis_name"])
        if thing["home_city"] == "LA":
            teams.add("Los Angeles" + " " + thing["home_name"])
        # sometimes team_city is different
        cities.add(thing["home_city"])
        cities.add(thing["vis_city"])
        players.update(thing["box_score"]["PLAYER_NAME"].values())
        cities.update(thing["box_score"]["TEAM_CITY"].values())
        if "LA" in cities:
            cities.add("Los Angeles")

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in suffixes:
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
    ignores = ["three point", "three - point", "three - pt", "three pt",
               "three- pointers", "three - pointer", "three pointers", "three - pointers"]
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


def preproc_text(text):
    """
    Replace abbrevations with their full identifier
    """
    for key, value in abbr2ent.items():
        text = text.replace(key, value)
    return text


def split_sent(sent):
    """
    Split sentence into words and replace number words with numbers
    """
    tokes = list()
    split_sent = preproc_text(sent).split(" ")
    i = 0
    while i < len(split_sent):
        # replace every number word with the corresponding digits
        if split_sent[i] in number_words and not annoying_number_word(split_sent, i):
            j = 1
            while i + j < len(split_sent) and split_sent in number_words and not annoying_number_word(split_sent,
                                                                                                      i + j):
                j += 1
            tokes.append(str(w2n.word_to_num(" ".join(split_sent[i:i + j]))))
            i += j
        else:
            tokes.append(split_sent[i])
            i += 1

    return tokes


def append_candidate_rels(entry, summ, prons, all_ents, players, teams, cities):
    """
    Appends tuples of form (sentence_tokens, [rels]) to candrels.
    """
    sents = sent_tokenize(summ)
    candrels = []
    for sent in sents:
        tokes = split_sent(sent)
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
        for entry in entries:
            summ = " ".join(entry['summary'])
            candrels.append(append_candidate_rels(entry, summ, prons, all_ents, players, teams, cities))

        extracted_stuff[corpus_type] = candrels

    del all_ents
    del players
    del teams
    del cities
    return extracted_stuff


def append_multilabeled_data(entry, sents, entdists, numdists, labels, vocab,
                             labeldict, max_len, len_entries, idx_list, idx):
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
        sent.extend([0] * (max_len - len(sent)))
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
        idx_list.append(idx)


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


def to_device(tensor_list):
    return [t.to(device, non_blocking=True) for t in tensor_list]
