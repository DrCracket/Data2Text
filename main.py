###############################################################################
# Control room to acces Information Extraction, Content Selection & Planning  #
# and Text Generation Module
###############################################################################

from extractor import get_extractor
from planner import get_planner

extractor = get_extractor()
get_planner(extractor)
