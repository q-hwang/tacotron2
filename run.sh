#! /bin/bash

out_file=${1}
python train.py --output_directory=${out_file} --log_directory=logdir
