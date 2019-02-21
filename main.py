###############################################################################
# Control room to acces Information Extraction, Content Selection & Planning  #
# and Text Generation Module                                                  #
###############################################################################

import logging
from extractor import get_extractor
from planner import get_planner, eval_planner
from util.generator import load_generator_data
from generator import get_generator

logging.basicConfig(level=logging.INFO)

extractor = get_extractor()
planner = get_planner(extractor)
generator = get_generator(extractor, planner)

# dataset = load_generator_data("train", extractor, planner)
# print(dataset[0])
# eval_planner(extractor, planner)
