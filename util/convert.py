###############################################################################
# Converts RotoWire json data into Markdown                                   #
# TODO: Add commandline support                                               #
# TODO: Update to use archive                                                 #
###############################################################################

import logging
from json import loads
from tabulate import tabulate
from random import randrange, choice


def getBoxScore(game):
    """Reads the boxscores from a json array and converts them to a table"""

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
    """Reads the linescores from a json array and converts them to a table"""

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
    """Reads everything from a json array that is not a boxscore or a
    linescore and converts it to a table"""

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


def getSummary(game):
    """Reads the summary from a json array"""

    summary = game["summary"]
    # insert a line break after each 40th word for prettier formatting
    summary = [word if idx == 0 or idx %
               40 != 0 else word + " \n" for idx, word in enumerate(summary)]

    # add a whitespace at the beginning for proper formatting
    return " " + " ".join(summary)


def genDescription(data, file_, index):
    """Selects an entry of the json database and saves it as a markdown
    string"""

    game = data[index]
    filename = "b_game_{}.md".format(str(index + 1))
    description = str()

    # print the title
    description += "# Basketball Game #{} from {}\n\n\n".format(
        str(index + 1), file_)

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
    description += "## Summary\n\n"
    description += getSummary(game)

    return description, filename


def RWData(path, file_, index=None):
    """reads the json database and saves a randomly selected entry as a
    markdown file"""

    with open(path + file_) as data:
        data = loads(data.read())

        if index is None:
            index = randrange(0, len(data))

        description, filename = genDescription(data, file_, index)

        with open(filename, "w") as file_:
            file_.write(description)


def findByCity(path, files, home_city, vis_city):

    for file_ in files:
        with open(path + file_) as data:
            data = loads(data.read())

        for index, game in enumerate(data):
            if game["home_city"] == home_city and game["vis_city"] == vis_city:
                description, filename = genDescription(data, file_, index)
                logging.info("Found match, saved to: " + filename)

                with open(filename, "w") as f:
                    f.write(description)


def findByName(path, files, home_name, vis_name):

    for file_ in files:
        with open(path + file_) as data:
            data = loads(data.read())

        for index, game in enumerate(data):
            if game["home_name"] == home_name and game["vis_name"] == vis_name:
                description, filename = genDescription(data, file_, index)
                logging.info("Found match, saved to: " + filename)

                with open(filename, "w") as f:
                    f.write(description)


databases = ["test.json", "train.json", "valid.json"]
database = choice(databases)
RWData("rotowire/", database, 1)
