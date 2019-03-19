#!/usr/bin/env python3
###############################################################################
# Control room to acces Information Extraction, Content Selection & Planning  #
# and Text Generation Module                                                  #
###############################################################################

import logging
import argparse
import datetime
from extractor import (train_extractor, eval_extractor, load_extractor,
                       extractor_is_available)
from planner import (train_planner, eval_planner, load_planner,
                     planner_is_available)
from generator import (train_generator, eval_generator, load_generator,
                       generator_is_available)
from util.pretty_print import genMdFiles


def get_extractor(parser, cnn):
    if extractor_is_available(cnn):
        extractor = load_extractor(cnn)
        return extractor
    else:
        parser.exit(status=1)


def get_planner(parser, cnn):
    if planner_is_available():
        extractor = get_extractor(parser, cnn)
        planner = load_planner(extractor)
        return planner
    else:
        parser.exit(status=1)


def get_generator(parser, cnn):
    if generator_is_available():
        extractor = get_extractor(parser, cnn)
        planner = get_planner(parser, cnn)
        generator = load_generator(extractor, planner)
        return generator
    else:
        parser.exit(status=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Generate descriptions for
            for data tables from the BoxScore dataset.""")
    parser.add_argument("--cnn",
                        action="store_true",
                        help="""Global setting that controls if the cnn
                        extractor should be used instead of the lstm
                        extractor.""")
    parser.add_argument("--no-log",
                        action="store_true",
                        help="Disable generation of log-files.")

    subparsers = parser.add_subparsers(dest="command")

    eval_parser = subparsers.add_parser("evaluate",
                                        help="""Evaluate a model of the text
                                        generation pipeline.""")
    eval_parser.add_argument("--stage",
                             choices=["extractor", "planner", "generator"],
                             default="generator",
                             help="Specify the model to evaluate.")
    eval_parser.add_argument("--corpus",
                             choices=["valid", "test"],
                             default="valid",
                             help="Specify the corpus to use for evaluation.")

    gen_parser = subparsers.add_parser("generate",
                                       help="""Generate the textual
                                       descriptions for a corpus. Results are
                                       saved in the 'generations'-folder.""")
    gen_parser.add_argument("--corpus",
                            choices=["train", "valid", "test"],
                            default="valid",
                            help="Specify the corpus to use for generation.")

    gen_parser = subparsers.add_parser("train",
                                       help="""Train the whole pipeline or a
                                       model from a specific pipeline
                                       stage.""")
    gen_parser.add_argument("--stage",
                            choices=["extractor", "planner", "generator",
                                     "pipeline"],
                            default="pipeline")

    args = parser.parse_args()

    # configure logging
    if not args.no_log:
        now = datetime.datetime.now()
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%m-%d %H:%M",
                            filename=now.strftime("%m-%d_%H:%M.log"),
                            filemode="w")
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    if args.command == "evaluate":
        corpus = args.corpus == "test"
        if args.stage == "extractor":
            extractor = get_extractor(parser, cnn=args.cnn)
            eval_extractor(extractor, test=corpus)
        elif args.stage == "planner":
            extractor = get_extractor(parser, cnn=args.cnn)
            planner = get_planner(parser, cnn=args.cnn)
            eval_planner(extractor, planner, test=corpus)
        elif args.stage == "generator":
            extractor = get_extractor(parser, cnn=args.cnn)
            planner = get_planner(parser, cnn=args.cnn)
            generator = get_generator(parser, cnn=args.cnn)
            eval_generator(extractor, planner, generator, test=corpus)

    elif args.command == "generate":
        extractor = get_extractor(parser, cnn=args.cnn)
        planner = get_planner(parser, cnn=args.cnn)
        generator = get_generator(parser, cnn=args.cnn)
        genMdFiles(extractor, planner, generator, args.corpus)

    elif args.command == "train":
        if args.stage == "extractor":
            train_extractor(cnn=args.cnn)
        elif args.stage == "planner":
            extractor = get_extractor(parser, cnn=args.cnn)
            train_planner(extractor)
        elif args.stage == "generator":
            extractor = get_extractor(parser, cnn=args.cnn)
            planner = get_planner(parser, cnn=args.cnn)
            train_generator(extractor, planner)
        elif args.stage == "pipeline":
            extractor = train_extractor(cnn=args.cnn)
            planner = train_planner(extractor)
            train_generator(extractor, planner)
