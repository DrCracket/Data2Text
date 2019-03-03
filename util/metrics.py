import tarfile
import torch
from json import loads
from abc import abstractmethod, ABC
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from .helper_funcs import get_ents, append_candidate_rels, append_multilabeled_data
from .extractor import load_extractor_data
from .constants import prons, ls_keys, bs_keys
from .data_structures import Vocab
from collections import OrderedDict


class ExtractiveMetric(ABC):
    """
    Base class for all extractive metrics
    """

    idx2type = None
    labeldict = None
    stats = None
    extractor = None
    vocab = None
    dataset = None
    entities = None

    def __init__(self, extractor, corpus_type, folder="boxscore-data", dataset="rotowire"):
        super().__init__()

        extractor_data = load_extractor_data(corpus_type, folder, dataset)
        self.idx2type = extractor_data.idx2type
        self.labeldict = dict(((v, k) for k, v in self.idx2type.items()))
        self.stats = extractor_data.stats
        self.extractor = extractor
        self.vocab = Vocab()
        for k, v in sorted(extractor_data.idx2word.items())[2:]:
            self.vocab[v] = k

        location = f"{folder}/{dataset}.tar.bz2"
        with tarfile.open(location, 'r:bz2') as f:
            self.dataset = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())
            self.entities = get_ents(self.dataset)

    def get_labels(self, candrel):
        """
        identify potential relations with the extractor from the candiate
        relations
        """
        sents, entdists, numdists = [], [], []
        max_len = max((len(tup[0]) for tup in candrel))
        append_multilabeled_data(candrel, sents, entdists, numdists, [],
                                 self.vocab, self.labeldict,
                                 max_len, [], [], -1)
        # create tensors from lists
        sents_ts = torch.tensor(sents)
        entdists_ts = torch.tensor(entdists)
        numdists_ts = torch.tensor(numdists)

        # clamp values so no other distances except the ones in the train set occur
        entdists_ts = entdists_ts.clamp(self.stats["min_entd"], self.stats["max_entd"])
        numdists_ts = numdists_ts.clamp(self.stats["min_numd"], self.stats["max_numd"])

        # shift values to eliminate negative numbers
        entdists_ts -= self.stats["min_entd"]
        numdists_ts -= self.stats["min_numd"]

        with torch.no_grad():
            self.extractor.eval()
            labels = self.extractor(sents_ts, entdists_ts, numdists_ts)

        return labels

    def get_record_tuples(self, entry, labels, candrel, resolve_record=False):
        """
        return relations which the extractor identified that really appear in
        the dataset
        """
        unique_rels = OrderedDict()
        extracted = list()
        for tup in candrel:
            for rel in tup[1]:
                ent, num, _, idthing = rel
                unique_rels[ent, num] = idthing
        for label_idx, ((ent, num), entry_idx) in enumerate(unique_rels.items()):
            label = self.idx2type[labels.argmax(dim=1)[label_idx].item()]
            if label != "NONE":
                if resolve_record:  # check if the record really exists in the dataset
                    if type(entry_idx) == bool and label in ls_keys:  # it's a team record
                        if entry_idx:  # it's the home team
                            if entry["home_line"][label] == str(num[2]):
                                team = entry["home_line"]["TEAM-CITY"] + " " + entry["home_line"]["TEAM-NAME"]
                                extracted.append((team, num[2], label))
                        else:  # it's the visitor team
                            if entry["vis_line"][label] == str(num[2]):
                                team = entry["vis_line"]["TEAM-CITY"] + " " + entry["vis_line"]["TEAM-NAME"]
                                extracted.append((team, num[2], label))
                    elif type(entry_idx) == str and label in bs_keys:  # if string, the record is a player record
                        # remove the "PLAYER-" prefix because the creators of the dataset didn't use that there
                        if entry["box_score"][label.replace("PLAYER-", "")][entry_idx] == str(num[2]):
                            player = entry["box_score"]["PLAYER_NAME"][entry_idx]
                            extracted.append((player, num[2], label))
                else:
                    # get an unambiguous identifier for each entity, for correct comparison
                    home_team = entry["home_line"]["TEAM-CITY"] + " " + entry["home_line"]["TEAM-NAME"]
                    vis_team = entry["vis_line"]["TEAM-CITY"] + " " + entry["vis_line"]["TEAM-NAME"]
                    entities = list(entry["box_score"]["PLAYER_NAME"].values())
                    entities.extend([home_team, vis_team])
                    matched_entity = max(((entity, len(set(entity.split()).intersection(ent[2].split())))
                                          for entity in entities), key=lambda word: word[1])
                    extracted.append((matched_entity[0], num[2], label))

        return extracted

    def extract_tuples(self, summary, idx, resolve_record=False):
        summary = " ".join(summary)
        entry = self.dataset[idx]
        extracted = list()
        candrel = append_candidate_rels(entry, summary, prons, *self.entities)
        if len(candrel) != 0:
            labels = self.get_labels(candrel)
            extracted = self.get_record_tuples(entry, labels, candrel, resolve_record)

        return extracted

    @abstractmethod
    def __call__(self):
        pass


class CSMetric(ExtractiveMetric):
    """
    Content Selection (CS): precision and re-call of unique relations extracted
    from the generated summary that are also extracted from the gold summary. This
    measures how well the generated document matches the gold document in terms of
    selecting which records to generate.
    """

    recall = 0
    precision = 0
    size = 0

    def __call__(self, gen_sum, gold_sum, sum_idx):
        """
        Calculate CS recall and precision:

                           | gen extracted | gen not extracted
        -------------------+---------------+------------------
        gold extracted     |      both     |    only_gold
        gold not extracted |    only_gen   |        -

        precision = both / (only_gen + both)
        recall = both / (only_gold + both)
        """
        gen_tuples = self.extract_tuples(gen_sum, sum_idx)
        gold_tuples = self.extract_tuples(gold_sum, sum_idx)

        both = len(set(gen_tuples) & set(gold_tuples))
        only_gen = len(set(gen_tuples) - set(gold_tuples))
        only_gold = len(set(gold_tuples) - set(gen_tuples))

        self.precision += both / (only_gen + both) if (only_gen + both) != 0 else 0
        self.recall += both / (only_gold + both) if (only_gold + both) != 0 else 0
        self.size += 1

    def calculate(self):
        """
        Return accumulated CS metrics
        """
        acc_prec = 100 * self.precision / self.size if self.size != 0 else 0
        acc_rec = 100 * self.recall / self.size if self.size != 0 else 0

        return acc_prec, acc_rec


class RGMetric(ExtractiveMetric):
    """
    Relation Generation (RG): precision and number of unique relations
    extracted from the generated text, that also appear in the dataset. This
    measures how well the system is able to generate text con-taining factual
    (i.e., correct) records.
    """

    number = 0
    precision = 0
    size = 0

    def __call__(self, gen_sum, sum_idx):
        """
        Calculate precision and number of extracted relations
        """
        extracted_records = set(self.extract_tuples(gen_sum, sum_idx))
        factual_records = set(self.extract_tuples(gen_sum, sum_idx, resolve_record=True))

        self.number += len(factual_records)
        self.precision += len(factual_records) / len(extracted_records) if len(extracted_records) != 0 else 0
        self.size += 1

    def calculate(self):
        """
        Return accumulated RG metrics
        """
        acc_prec = 100 * self.precision / self.size if self.size != 0 else 0
        acc_num = self.number / self.size if self.size != 0 else 0

        return acc_prec, acc_num


class COMetric(ExtractiveMetric):
    """
    Content Ordering (CO): normalized Damerau-Levenshtein Distance (Brill and
    Moore,2000) between the sequences of records extracted from the gold
    summary and that extracted from the generated summary. This measures how
    well the system orders the records it chooses to discuss.
    """

    dld = 0
    size = 0

    def __call__(self, gen_sum, gold_sum, sum_idx):
        """
        Calculate normalized Damerau-Levenshtein Distance between 2 summaries.
        Substitute ich record tuple with a character.
        Assumes that all tuples per summary are unique.
        """
        gen_records = self.extract_tuples(gen_sum, sum_idx)
        gold_records = self.extract_tuples(gold_sum, sum_idx)

        ascii_start = 0
        gen_idents = ''.join((chr(ascii_start + i) for i in range(len(gen_records))))
        gold_idents = ''

        for record in gold_records:
            try:
                gold_idents += gen_idents[gen_records.index(record)]
            except ValueError:
                gold_idents += chr(len(gen_idents) + len(gold_idents))

        self.dld += 1 - normalized_damerau_levenshtein_distance(gen_idents, gold_idents)
        self.size += 1

    def calculate(self):
        """
        Return accumulated CO metrics
        """
        acc_dld = 100 * self.dld / self.size if self.size != 0 else 0

        return acc_dld
