###############################################################################
# Control room to acces Information Extraction, Content Selection & Planning  #
# and Text Generation Module                                                  #
###############################################################################

from extractor import get_extractor
from planner import get_planner, eval_planner
from data_utils import load_generator_data
from generator import get_generator

extractor = get_extractor()
planner = get_planner(extractor)
generator = get_generator(extractor, planner)

# dataset = load_generator_data("train", extractor, planner)
# print(dataset[0])
# eval_planner(extractor, planner)
