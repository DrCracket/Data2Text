###############################################################################
# Converts RotoWire json data into Markdown                                   #
# TODO: Add commandline support                                               #
# TODO: Update to use archive                                                 #
###############################################################################

import tarfile
from json import loads
from tabulate import tabulate
from random import randrange
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .generator import load_generator_data, generate_text


def getBoxScore(game):
    """
    Reads the boxscores from a json array and converts them to a table
    """
    header = list()
    table = list()

    # bs_value is a dictionary with entries for each player for property bs_key
    for bs_key, bs_value in game["box_score"].items():
        table_column = list()
        # sort the entries numerically => convert strings to integers
        for key, value in sorted(bs_value.items(), key=lambda k: int(k[0])):
            table_column.append(value)
        # the player names should be the first column
        if bs_key == "PLAYER_NAME":
            header.insert(0, bs_key)
            table.insert(0, table_column)
        else:
            header.append(bs_key)
            table.append(table_column)

    # transpose the table so that each row represents a player
    table = map(list, zip(*table))
    return tabulate(table, header, tablefmt="pipe", numalign="center",
                    stralign="center")


def getLineScore(game, type_):
    """
    Reads the linescores from a json array and converts them to a table
    """
    header = list()
    table = list()

    # there are two types of linescores, corresponding to home & visitor teams
    for key, value in game[type_].items():
        # the team name should be the first column
        if key == "TEAM-NAME":
            header.insert(0, key)
            table.insert(0, [value])
        else:
            header.append(key)
            table.append([value])

    # transpose the table so that each row represents a team
    table = map(list, zip(*table))
    return tabulate(table, header, tablefmt="pipe", numalign="center",
                    stralign="center")


def getOtherInfo(game):
    """
    Reads everything from a json array that is not a boxscore or a linescore
    and converts it to a table
    """
    header = list()
    table = list()

    for key in game:
        if key not in ["box_score", "home_line", "vis_line", "summary"]:
            header.append(key)
            table.append([game[key]])

    # transpose the table
    table = map(list, zip(*table))
    return tabulate(table, header, tablefmt="pipe", numalign="center",
                    stralign="center")


def getSummary(game, gen_summary):
    """
    Reads the summary from a json array and tokenizes it
    Does the same thing for the generated summary
    """
    detokenizer = TreebankWordDetokenizer()
    summary = game["summary"]

    return detokenizer.detokenize(summary), detokenizer.detokenize(gen_summary)


def genDescription(game, corpus_type, index, gen_summary):
    """
    Selects an entry of the json database and saves it as a markdown string
    """
    filename = "b_game_{}.md".format(str(index + 1))
    description = str()

    # print the title
    description += "# Basketball Game #{} from {} corpus\n\n\n".format(
        str(index + 1), corpus_type)

    # box-scores
    description += "## Box-Scores\n\n"
    description += getBoxScore(game) + "\n\n\n"

    # line-scores
    description += "## Line-Scores for Home Team\n\n"
    description += getLineScore(game, "home_line") + "\n\n\n"

    description += "## Line-Scores for Visitor Team\n\n"
    description += getLineScore(game, "vis_line") + "\n\n\n"

    # other information
    description += "## Other Information\n\n"
    description += getOtherInfo(game) + "\n\n\n"

    # summary
    gold_summary, gen_summary = getSummary(game, gen_summary)
    description += "## Gold Summary\n\n"
    description += gold_summary + "\n\n\n"
    description += "## Generated Summary\n\n"
    description += gen_summary

    return description, filename


def genMdFile(extractor, planner, generator, corpus_type, index=None,
              folder="boxscore-data", dataset="rotowire"):
    """
    Reads the json database and saves a randomly selected entry as a
    markdown file, if the index is not specified
    """
    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_data = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())
    gen_data = load_generator_data(corpus_type,
                                   extractor,
                                   planner,
                                   folder,
                                   dataset)

    if index is None:
        index = randrange(0, len(gen_data))

    entry = gen_data[index]
    roto_index = gen_data.idx_list[index]
    gen_summary = generate_text(generator,
                                gen_data.vocab,
                                gen_data.idx2word,
                                entry)

    description, filename = genDescription(raw_data[roto_index],
                                           corpus_type,
                                           roto_index,
                                           gen_summary)

    with open(filename, "w") as file_:
        file_.write(description)
