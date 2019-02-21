###############################################################################
# Various Constants used throughout the network                               #
###############################################################################

HOME = "HOME"
AWAY = "AWAY"

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

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

# there shouldn't be longer content plans than 50 records
MAX_CONTENT_PLAN_LENGTH = 50
