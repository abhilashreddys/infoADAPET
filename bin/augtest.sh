#!/usr/bin/env bash
# set -exu
exp_dir=$1
# echo $exp_dir
python -m src.dev -e $exp_dir -d altername
python -m src.dev -e $exp_dir -d ant_mod
python -m src.dev -e $exp_dir -d charal
python -m src.dev -e $exp_dir -d loc_mod
python -m src.dev -e $exp_dir -d num_mod
python -m src.dev -e $exp_dir -d paraphr
python -m src.dev -e $exp_dir -d pertub_neg
python -m src.dev -e $exp_dir -d syn_mod