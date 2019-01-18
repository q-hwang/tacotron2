#!/usr/bin/env python
# coding: utf-8

'''

    python inference [text] [checkpoint_path] [outdir] [name]

'''

# ## Tacotron 2 inference code
import sys, os
import numpy as np
import torch
from scipy.io.wavfile import write
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


def get_model(hparams, checkpoint_path):
    
    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass
    
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
       
    model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
    # model.cpu()
    print("evaluating...")
    _ = model.eval()
    print("done")
    return model

def get_input(text):
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
      torch.from_numpy(sequence)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).long()
    return sequence



def _normalize(S):
    return np.clip((S + 100) / 100, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * 100) - 100

def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def to_wavenet_mel(mel_np):
    axis = np.linspace(t.hz_to_mel(75), t.hz_to_mel(7600),80)
    trimed = np.zeros(mel_np.shape)
    for i in range(mel_np.shape[0]):
        for j in range(mel_np.shape[1]):
            trimed[i][j] = mel_np[i][(int)(axis[j]*80/t.hz_to_mel(11025))]

    wavenet_mel = _normalize(_amp_to_db(np.exp(trimed) ) - 20)
    return wavenet_mel


def main(text, checkpoint_path, path, name):
    #### Setup hparams
    hparams = create_hparams("distributed_run=False,mask_padding=False")
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    
    #### Load model from checkpoint
    model = get_model(hparams,checkpoint_path)

    #### Prepare text input
    sequence = get_input(get_pinyin(text))

    #### inference
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, drop_prob=0.25)
   
    #### tacotron result
    taco_stft = TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length, 
    sampling_rate=hparams.sampling_rate)
    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), 
                       taco_stft.stft_fn, 60)
    write(os.path.join(path, name) + '_tacotron.wav', 16000, waveform[0].data.cpu().numpy())

    #### transform tacotron mel to wavenet mel
    wavenet_mel = to_wavenet_mel(mel_outputs_postnet.data.cpu().numpy()[0].T)



    #### save
    np.save(os.path.join(path, name) + '_mel.npy',mel_outputs_postnet.data.cpu().numpy()[0])
    np.save(os.path.join(path, name) + '_alig.npy',alignments.data.cpu().numpy()[0])
    np.save(os.path.join(path, name) + '.npy',wavenet_mel)


if __name__ == '__main__':
    
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    