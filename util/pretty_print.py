###############################################################################
# Converts game records, gold summaries and generated summaries into the      #
# easiliy readable and convertible markdown format.                           #
###############################################################################

import tarfile
from json import loads
from tabulate import tabulate
from os import path, makedirs
from nltk.tokenize.treebank import TreebankWordDetokenizer
from .generator import load_generator_data, TextGeneratorWrapper
from .planner import load_planner_data, content_plan_to_text
from .metrics import CSMetric, RGMetric, COMetric, BleuScore


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


def getContentPlanTable(content_plan):
    """
    Bring a content plan string array in table format
    """
    headers = ["Entity", "Type", "Value", "Home or Away"]
    return tabulate(content_plan, headers, tablefmt="pipe", numalign="center",
                    stralign="center")


def getSummary(game, gen_summary):
    """
    Reads the summary from a json array and tokenizes it
    Does the same thing for the generated summary
    """
    detokenizer = TreebankWordDetokenizer()
    summary = game["summary"]

    return detokenizer.detokenize(summary), detokenizer.detokenize(gen_summary)


def genDescription(game, corpus_type, index, gen_summary, cont_plan, metrics,
                   planner=False):
    """
    Selects an entry of the json database and saves it as a markdown string
    """
    prefix = "" if planner else "template_"
    filename = "{}{}_{}.md".format(prefix, corpus_type, str(index + 1))
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
    description += gen_summary + "\n\n\n"

    # extractive metrics + bleu
    description += "## Metrics\n\n"
    description += "* **CS Precision:** {:.4f}%, CS Recall: {:.4f}%" \
                   .format(*metrics["CS"]) + "\n"
    description += "* **RG Precision:** {:.4f}%, RG #: {:.4f}" \
                   .format(*metrics["RG"]) + "\n"
    description += "* **CO Damerau-Levenshtein Distance:** {:.4f}%" \
                   .format(metrics["CO"]) + "\n"
    description += "* **BLEU Score:** {:.4f}" \
                   .format(metrics["BLEU"]) + "\n\n\n"

    # content plan
    description += "## Content Plan\n\n"
    description += getContentPlanTable(cont_plan)

    return description, filename


def calculate_metrics(cs_metric, rg_metric, co_metric, bleu_metric,
                      extractor, roto_index, gen_sum, gold_sum, corpus_type):
    """
    Calculate all extractive metrics plus BLEU score
    """
    cs_metric.clear()
    rg_metric.clear()
    co_metric.clear()
    bleu_metric.clear()

    cs_metric(gen_sum, gold_sum, roto_index)
    rg_metric(gen_sum, roto_index)
    co_metric(gen_sum, gold_sum, roto_index)
    bleu_metric(gen_sum, gold_sum)

    metrics = {"CS": cs_metric.calculate(),
               "RG": rg_metric.calculate(),
               "CO": co_metric.calculate(),
               "BLEU": bleu_metric.calculate()
               }

    return metrics


def genMdFiles(extractor, content_planner, generator, corpus_type, value=None,
               folder="boxscore-data", dataset="rotowire", planner=False):
    """
    Generates text for all dataset entries of a corpus and saves them as
    markdown files.
    """
    with tarfile.open(f"{folder}/{dataset}.tar.bz2", "r:bz2") as f:
        raw_data = loads(f.extractfile(f"{dataset}/{corpus_type}.json").read())
    gen_data = load_generator_data(corpus_type,
                                   extractor,
                                   content_planner,
                                   folder,
                                   dataset,
                                   planner)
    plan_data = load_planner_data(corpus_type,
                                  extractor,
                                  folder,
                                  dataset)

    t_generator = TextGeneratorWrapper(generator, gen_data)
    cs_metric = CSMetric(extractor, corpus_type)
    rg_metric = RGMetric(extractor, corpus_type)
    co_metric = COMetric(extractor, corpus_type)
    bleu_metric = BleuScore()

    if value is None:
        iterable = range(0, len(gen_data))
    else:
        iterable = [value]

    for index in iterable:
        roto_index = gen_data.idx_list[index]
        gen_summary_markup, gen_summary = t_generator.generate_text(index)
        content_plan_str = content_plan_to_text(plan_data.sequence[index],
                                                gen_data.content_plan[index],
                                                plan_data.idx2word)
        metrics = calculate_metrics(cs_metric, rg_metric, co_metric,
                                    bleu_metric, extractor, roto_index,
                                    gen_summary,
                                    raw_data[roto_index]["summary"],
                                    corpus_type)
        description, filename = genDescription(raw_data[roto_index],
                                               corpus_type,
                                               roto_index,
                                               gen_summary_markup,
                                               content_plan_str,
                                               metrics,
                                               planner)
        if not path.exists("generations"):
            makedirs("generations")
        with open("generations/" + filename, "w") as file_:
            file_.write(description)
