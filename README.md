# Data-to-Text Generation
This repository contains an implementation based on the models described in
[Data-to-Text Generation with Content Selection and Planning](https://arxiv.org/pdf/1809.00582.pdf).
This project also contains a neural content planner to produce content plans,
however by default this implementation uses template-based content plans, which improve
the generated texts.

## Cloning
Clone the repository with the following command:
```sh
git clone --recurse-submodules https://git.ukp.informatik.tu-darmstadt.de/belouadi/data-to-text-generator.git
```

## Setup
For a quick start download the
[preprocessed dataset](https://www.dropbox.com/s/u3mjbwxskhfyg4m/data.zip?dl=0)
and the
[pretrained models](https://www.dropbox.com/s/8qhrocemwsz54jt/models.zip?dl=0)
and drop their contents into the root folder of this repository. This will save
you from preprocessing and training, which requires a lot of time.


Make sure that [pipenv](https://pipenv.readthedocs.io/en/latest/) is installed:
```sh
pip install pipenv
```
Then cd into the repository and create a virtual environment with the required
dependencies:
```sh
cd $PATH_TO_REPOSITORY
pipenv install
```
After that start a shell within the created virtual environment:
```sh
pipenv shell
```

## Text Generation
Generate the game summaries with the following command:
```sh
# could also be 'train' or 'test'
CORPUS=valid

./data2text.py generate --corpus $CORPUS

# if you want to use content plans created by the content-planner, use this command:
./data2text.py generate --corpus $CORPUS --use-planner
```
The generated texts will be saved as markdown files in the *generations*
folder. Every markdown file contains the generated summary, the gold summary,
the associated records, information on which values where copied, the content
plan and the metrics.

If you want to compare the texts according to their metrics, you can use the
*sort_by.sh* script:
```sh
# could also be 'co_distance', 'rg_precision', 'rg_number', 'cs_precision' or 'cs_recall'   
METRIC=bleu

./sort_by.sh $METRIC        
```

## Evaluation
Every step in the model pipeline can be evaluated with the following command:
```sh
# could also be 'extractor' or 'planner'
STAGE=generator
# could also be test
CORPUS=valid

./data2text.py evaluate --stage $STAGE --corpus $CORPUS

# if you want to use content plans created by the content-planner, use this command:
./data2text.py evaluate --stage $STAGE --corpus $CORPUS --use-planner
```

## Training
If you want to train the models yourself, you can do so with the following
command:
```sh
# could also be 'extractor', 'planner' or 'generator'
STAGE=pipeline

./data2text.py train --stage $STAGE
```

## Misc
For advanced usage check out the *help* argument:
```sh
./data2text.py --help
```
