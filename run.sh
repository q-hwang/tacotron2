#! /bin/bash

out_file=${1}
data_dir=${2}
python train.py --output_directory=${out_file} --log_directory=logdir --hparams=training_files=${data_dir}/train.txt,validation_files=${data_dir}/eval.txt
