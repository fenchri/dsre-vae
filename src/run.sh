#!/bin/bash

#python main.py --config ../configs/wikidistant_base.yaml --mode train
#python main.py --config ../configs/wikidistant_reco.yaml --mode train
#python main.py --config ../configs/wikidistant_prior.yaml --mode train
python main.py --config ../configs/nyt10_base.yaml --mode train
python main.py --config ../configs/nyt10_reco.yaml --mode train
python main.py --config ../configs/nyt10_prior.yaml --mode train
