###############################################################################
# Control room to acces Information Extraction, Content Selection & Planning  #
# and Text Generation Module                                                  #
###############################################################################

import logging
from extractor import get_extractor
from planner import get_planner, eval_planner
from util.generator import load_generator_data
from generator import get_generator
from util.pretty_print import genMdFile

logging.basicConfig(level=logging.INFO)

extractor = get_extractor(lstm=True)
planner = get_planner(extractor)
generator = get_generator(extractor, planner)

genMdFile(extractor, planner, generator, "valid")
# dataset = load_generator_data("train", extractor, planner)
# print(dataset[0])
# eval_planner(extractor, planner)
