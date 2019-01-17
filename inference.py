#!/usr/bin/env python
# coding: utf-8

'''

    python inference [checkpoint_path] [text] [outdir] [name]

'''

# ## Tacotron 2 inference code
import sys, os
import numpy as np
import torch
import librosa.core.time_frequency as t

from data_utils import TextMelLoader
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from synthesis import synthesis_griffin_lim

#TODO: fix dependency
from chinese_process import get_pinyin


def get_model(checkpoint_path):
    #### Setup hparams
    hparams = create_hparams("distributed_run=False,mask_padding=False")
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(checkpoint_path)['state_dict'].items()})
    # model.cpu()
    print("evaluating...")
    _ = model.eval()
    print("done")
    return model

def get_input(text):
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
      torch.from_numpy(sequence)).cuda().long()
    return sequence



def _normalize(S):
    return np.clip((S + 100) / 100, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * 100) - 100

def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def to_wavenet_mel(tacotron_mel):
    axis = np.linspace(t.hz_to_mel(75), t.hz_to_mel(7600),80)
    trimed = np.zeros(mel_np.shape)
    for i in range(mel_np.shape[0]):
        for j in range(mel_np.shape[1]):
            trimed[i][j] = mel_np[i][(int)(axis[j]*80/t.hz_to_mel(11025))]

    wavenet_mel = _normalize(_amp_to_db(np.exp(trimed) ) - 20)
    return wavenet_mel


if __name__ == '__main__':

    #### Load model from checkpoint
    checkpoint_path = sys.argv[1]
    # checkpoint_path = "/home/guandao/Projects/tacotron2/jan13_p=0.25/checkpoint_114000"
    model = get_model(checkpoint_path)

    #### Prepare text input
    sequence = get_input(get_pinyin(sys.argv[2]))

    #### inference
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, drop_prob=0.25)


    #### transform tacotron mel to wavenet mel
    wavenet_mel = to_wavenet_mel(mel_outputs_postnet.data.cpu().numpy()[0].T)



    #### save
    path = sys.argv[3]
    name = sys.argv[4]
    np.save(os.path.join(path, name) + '_mel.npy',mel_outputs_postnet.data.cpu().numpy()[0])
    np.save(os.path.join(path, name) + '_alig.npy',alignments.data.cpu().numpy()[0])
    np.save(os.path.join(path, name) + '.npy',wavenet_mel)


