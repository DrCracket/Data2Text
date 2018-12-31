###############################################################################
#                  Loader for RotoWire dataset                                #
###############################################################################

from json import loads
import torch
import tarfile


class load:
    def __init__(self, file_, folder="boxscore-data", dataset="rotowire"):
        path = folder + "/" + dataset + ".tar.bz2"
        member = dataset + "/" + file_

        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.summaries = list()
        self.n_words = 0
        # Count SOS, EOS, (H)ome and (V)isitor Team
        for word in ["PADDING", "SOS", "EOS", "H", "V"]:
            self.checkWord(word)
        dim1 = 0
        dim2 = 0
        with tarfile.open(path, "r:bz2") as data:
            dataset = loads(data.extractfile(member).read())
            dim1 = len(dataset)
            for sample in dataset:
                dim2 = self.getSmpLen(sample) if self.getSmpLen(
                    sample) > dim2 else dim2
            self.samples = torch.zeros(dim1, dim2, 4, dtype=torch.long)
            for idx_dim1, sample in enumerate(dataset):
                self.addSample(sample, idx_dim1)
                break

    def getSmpLen(self, sample):
        # substract the elements that don't get added as records
        # (e.g. because they are used as keys or are ignored)
        length = -8-len(sample["box_score"]["PLAYER_NAME"])
        for val in sample.values():
            if isinstance(val, dict):
                for val in val.values():
                    if isinstance(val, dict):
                        length += len(val)
                    else:
                        length += 1
            else:
                length += 1
        return length

    def addSample(self, sample, idx_dim1):
        box_score = sample["box_score"]
        line_scores = [sample["home_line"], sample["vis_line"]]
        home_team = sample["home_city"]

        # it seems that they are not included according to the paper
        # other_scores = {x: sample[x] for x in sample if x not in ["box_score",
        #    "home_line", "vis_line", "summary"]}
        # other_scores["day"] = datetime.strptime(other_scores["day"],
        #        "%m_%d_%y").strftime('%A')
        # other_scores["GAME"] =  "game"
        # entities.extend(self.addScore(other_scores, "GAME"))

        self.addScores(box_score, line_scores, home_team, idx_dim1)

        self.summaries.append(self.addSummary(sample["summary"]))

    def addScores(self, boxscore, linescores, home_team, idx_dim1):
        idx_dim2 = 0
        # parse boxscores
        for type_, bs_value in boxscore.items():
            if type_ != "PLAYER_NAME":  # don't use PLAYER_NAME as a value in a record, because of the tokenisation it would never be used
                for key, value in bs_value.items():
                    is_home = boxscore["TEAM_CITY"][key] == home_team
                    self.addTuple(type_, boxscore["PLAYER_NAME"][key], value,
                                  is_home, idx_dim1, idx_dim2)
                    idx_dim2 += 1
        # parse linescores
        for linescore in linescores:
            for type_, value in linescore.items():
                is_home = linescore["TEAM-CITY"] == home_team
                self.addTuple(type_, linescore["TEAM-NAME"], value,
                              is_home, idx_dim1, idx_dim2)
                idx_dim2 += 1

    def addTuple(self, type_, entity, value, is_home, idx_dim1, idx_dim2):
        team = self.word2index["H"] if is_home else self.word2index["V"]
        triple = (type_, entity, value)
        for idx_dim3, word in enumerate(triple):
            self.checkWord(word.lower())
            self.samples[idx_dim1][idx_dim2][idx_dim3] = self.word2index[word.lower()]
        self.samples[idx_dim1][idx_dim2][3] = team

    def checkWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSummary(self, summary):
        for word in summary:
            self.checkWord(word.lower())
        return [self.word2index[word.lower()] for word in summary]
