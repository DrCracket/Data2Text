import tarfile
from json import loads
from nltk import sent_tokenize, word_tokenize
from word2number import w2n


class loader:
    prons = set(["he", "He", "him", "Him", "his", "His", "they",
                 "They", "them", "Them", "their", "Their"])  # leave out "it"
    singular_prons = set(["he", "He", "him", "Him", "his", "His"])
    plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

    number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])

    bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
               "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
               "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
               "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
               "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

    ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
               "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
               "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

    NUM_PLAYERS = 13

    def __init__(self, file_, folder="boxscore-data", dataset="rotowire"):
        path = folder + "/" + dataset + ".tar.bz2"
        member = dataset + "/" + file_

        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        self.summaries = list()

        with tarfile.open(path, 'r:bz2') as f:
            self.trdata = loads(f.extractfile(member).read())

        # Count SOS, EOS, (H)ome and (V)isitor Team
        for word in ["PAD", "SOS", "EOS", "H", "V"]:
            self.handle_word(word)

        self.max_sent_len = self.parse_summaries()
        records = self.make_records()
        content_plans = self.make_pointerfi()

    def handle_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        return self.word2index[word]

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
                        # print "replacing", sent[i], "with", referent[2], "in", " ".join(sent)
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
        # print sent
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
                # print "picking", bs["PLAYER_NAME"][keys[0]]
        if len(keys) == 0:
            for k, v in bs["FIRST_NAME"].items():
                if entname == v:
                    keys.append(k)
            if len(keys) > 1:  # if we matched on first name and there are a bunch just forget about it
                return None
        # if len(keys) == 0:
            # print "Couldn't find", entname, "in", bs["PLAYER_NAME"].values()
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

    def box_preproc2(self):
        """
        just gets src for now
        """
        srcs = [[] for i in range(2 * self.NUM_PLAYERS + 2)]

        for entry in self.trdata:
            home_players, vis_players = self.get_player_idxs(entry)
            for ii, player_list in enumerate([home_players, vis_players]):
                for j in range(self.NUM_PLAYERS):
                    src_j = []
                    player_key = player_list[j] if j < len(
                        player_list) else None
                    for k, key in enumerate(self.bs_keys):
                        rulkey = key.split('-')[1]
                        val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
                        src_j.append(val)
                    srcs[ii * self.NUM_PLAYERS + j].append(src_j)

            home_src, vis_src = [], []
            # required for now, so that linescores can be processed together with boxscores
            for k in range(len(self.bs_keys) - len(self.ls_keys)):
                home_src.append("PAD")
                vis_src.append("PAD")

            for k, key in enumerate(self.ls_keys):
                home_src.append(entry["home_line"][key])
                vis_src.append(entry["vis_line"][key])

            srcs[-2].append(home_src)
            srcs[-1].append(vis_src)

        return srcs

    def linearized_preproc(self, srcs):
        """
        maps from a num-rows length list of lists of ntrain to an
        ntrain-length list of concatenated rows
        """
        lsrcs = []
        for i in range(len(srcs[0])):
            src_i = []
            for j in range(len(srcs)):
                # slice list because we ignore first value and remove padding as it isn't needed anymore
                src_i.extend(list(filter(lambda w: w != "PAD", srcs[j][i][1:])))
            lsrcs.append(src_i)
        return lsrcs

    def fix_target_idx(self, summ, assumed_idx, word, neighborhood=5):
        """
        tokenization can mess stuff up, so look around
        """
        for i in range(1, neighborhood + 1):
            if assumed_idx + i < len(summ) and summ[assumed_idx + i] == word:
                return assumed_idx + i
            elif assumed_idx - i >= 0 and assumed_idx - i < len(summ) and summ[assumed_idx - i] == word:
                return assumed_idx - i
        return None

    def make_records(self):
        srcs = self.box_preproc2()
        lsrcs = []
        bs_len = len(self.bs_keys)
        records = list()

        # maps from a num-rows length list of lists of ntrain to an ntrain-length list of concatenated rows
        for i in range(len(srcs[0])):
            src_i = []
            for j in range(len(srcs)):
                src_i.extend(srcs[j][i])
            lsrcs.append(src_i)

        for sample in lsrcs:
            row = list()
            for i in range(0, len(sample), bs_len):
                entity = sample[i]
                values = sample[i + 1:i + bs_len]

                if entity != "PAD":  # parse as boxscore data
                    for value, type_ in zip(values, self.bs_keys[1:]):  # don't use PLAYER_NAME as value
                        row.append((self.handle_word(type_), self.handle_word(entity), self.handle_word(value)))
                else:  # parse linescore data
                    entity = sample[i + bs_len - 1]  # TEAM-NAME
                    values = sample[i + bs_len - len(self.ls_keys):i + bs_len + 1]  # ignore the padded entries
                    for value, type_ in zip(values, self.ls_keys):
                        row.append((self.handle_word(type_), self.handle_word(entity), self.handle_word(value)))
            records.append(row)

        return records

    def parse_summaries(self):
        max_sent_len = 0
        for entry in self.trdata:
            summ = entry["summary"]
            max_sent_len = len(summ) if len(summ) > max_sent_len else max_sent_len
            for word in summ:
                self.handle_word(word)

        return max_sent_len

    def make_content_plan(self, resolve_prons=False):
        """
        for each target word want to know where it could've been copied from
        N.B. this function only looks at string equality in determining pointerness.
        this means that if we sneak in pronoun strings as their referents, we won't point to the
        pronoun if the referent appears in the table; we may use this tho to point to the correct number
        """

        bs_len = (2 * self.NUM_PLAYERS) * (len(self.bs_keys) - 1)  # -1 b/c we trim first col
        ls_len = len(self.ls_keys)
        rulsrcs = self.linearized_preproc(self.box_preproc2())

        all_ents, players, teams, cities = self.get_ents(self.trdata)

        skipped = 0

        train_links = []
        for i, entry in enumerate(self.trdata):
            home_players, vis_players = self.get_player_idxs(entry)
            inv_home_players = {pkey: jj for jj,
                                pkey in enumerate(home_players)}
            inv_vis_players = {pkey: (jj + self.NUM_PLAYERS)
                               for jj, pkey in enumerate(vis_players)}
            summ = " ".join(entry['summary'])
            sents = sent_tokenize(summ)
            words_so_far = 0
            links = []
            prev_ents = []
            for j, sent in enumerate(sents):
                # just assuming this gives me back original tokenization
                tokes = word_tokenize(sent)
                ents = self.extract_entities(tokes, all_ents, self.prons, prev_ents, resolve_prons, players, teams, cities)
                if resolve_prons:
                    prev_ents.append(ents)
                nums = self.extract_numbers(tokes)
                # should return a list of (enttup, numtup, rel-name, identifier) for each rel licensed by the table
                rels = self.get_rels(entry, ents, nums, players, teams, cities)
                for (enttup, numtup, label, idthing) in rels:
                    if label != 'NONE':
                        # try to find corresponding words (for both ents and nums)
                        ent_start, ent_end, entspan, _ = enttup
                        num_start, num_end, numspan = numtup
                        if isinstance(idthing, bool):  # city or team
                            # get entity indices if any
                            for k, word in enumerate(tokes[ent_start:ent_end]):
                                src_idx = None
                                if word == entry["home_name"]:
                                    src_idx = bs_len + ls_len - 1  # last thing
                                elif word == entry["home_city"]:
                                    src_idx = bs_len + ls_len - 2  # second to last thing
                                elif word == entry["vis_name"]:
                                    src_idx = bs_len + 2 * ls_len - 1  # last thing
                                elif word == entry["vis_city"]:
                                    src_idx = bs_len + 2 * ls_len - 2  # second to last thing
                                if src_idx is not None:
                                    targ_idx = words_so_far + ent_start + k
                                    if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                        targ_idx = self.fix_target_idx(entry["summary"], targ_idx, word)
                                    # print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + ent_start + k]
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        # src_idx, target_idx
                                        links.append((src_idx, targ_idx))

                            # get num indices if any
                            for k, word in enumerate(tokes[num_start:num_end]):
                                src_idx = None
                                if idthing:  # home, so look in the home row
                                    if entry["home_line"][label] == word:
                                        col_idx = self.ls_keys.index(label)
                                        src_idx = bs_len + col_idx
                                else:
                                    if entry["vis_line"][label] == word:
                                        col_idx = self.ls_keys.index(label)
                                        src_idx = bs_len + ls_len + col_idx
                                if src_idx is not None:
                                    targ_idx = words_so_far + num_start + k
                                    if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                        targ_idx = self.fix_target_idx(entry["summary"], targ_idx, word)
                                    # print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k]
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        links.append((src_idx, targ_idx))
                        else:  # players
                            # get row corresponding to this player
                            player_row = None
                            if idthing in inv_home_players:
                                player_row = inv_home_players[idthing]
                            elif idthing in inv_vis_players:
                                player_row = inv_vis_players[idthing]
                            if player_row is not None:
                                # ent links
                                for k, word in enumerate(tokes[ent_start:ent_end]):
                                    src_idx = None
                                    if word == entry["box_score"]["FIRST_NAME"][idthing]:
                                        src_idx = (player_row + 1) * (len(self.bs_keys) - 1) - 2  # second to last thing
                                    elif word == entry["box_score"]["SECOND_NAME"][idthing]:
                                        src_idx = (player_row + 1) * (len(self.bs_keys) - 1) - 1  # last thing
                                    if src_idx is not None:
                                        targ_idx = words_so_far + ent_start + k
                                        if entry["summary"][targ_idx] != word:
                                            targ_idx = self.fix_target_idx(entry["summary"], targ_idx, word)
                                        if targ_idx is None:
                                            skipped += 1
                                        else:
                                            assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                            # src_idx, target_idx
                                            links.append((src_idx, targ_idx))
                                # num links
                                for k, word in enumerate(tokes[num_start:num_end]):
                                    src_idx = None
                                    if word == entry["box_score"][label.split('-')[1]][idthing]:
                                        # subtract 1 because we ignore first col
                                        src_idx = player_row * (len(self.bs_keys) - 1) + self.bs_keys.index(label) - 1
                                    if src_idx is not None:
                                        targ_idx = words_so_far + num_start + k
                                        if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                            targ_idx = self.fix_target_idx(entry["summary"], targ_idx, word)
                                        # print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k]
                                        if targ_idx is None:
                                            skipped += 1
                                        else:
                                            assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                            links.append((src_idx, targ_idx))

                words_so_far += len(tokes)
            train_links.append(links)
        print("SKIPPED", skipped)

        return train_links


loader = loader("train.json")
